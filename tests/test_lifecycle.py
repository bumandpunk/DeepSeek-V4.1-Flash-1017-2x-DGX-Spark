"""Failure boundaries for cluster lifecycle; no SSH, Docker or systemd is invoked."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import cluster,node

class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.config=json.loads((cluster.ROOT/'config/cluster.example.json').read_text())
        for n in self.config['nodes']:
            n['state']=self.temp.name+'/state'+str(n['rank']);n['cache']=self.temp.name+'/cache'+str(n['rank'])

    def test_invalid_paths_and_disabled_memory_guards_fail(self):
        for changes in [('cache','/srv/models/deepseek-1017/model/cache'),('state','/'),('key_file','relative.key')]:
            c=copy.deepcopy(self.config);c['nodes'][0][changes[0]]=changes[1]
            with self.assertRaises(ValueError):cluster.validate(c)
        c=copy.deepcopy(self.config);c['guard_floor_gib']=0
        with self.assertRaises(ValueError):cluster.validate(c)

    def test_profiles_do_not_combine_unvalidated_routes(self):
        for name,p in node.PROFILES.items():
            self.assertFalse(p['SPARK1017_LOOKUP']=='1' and p['SPARK1017_RDMA_POST']!='0')
        self.assertEqual(node.PROFILES['baseline'],{'SPARK1017_LOOKUP':'0','SPARK1017_RDMA_POST':'0'})

    def test_engine_cache_budget_keeps_margin_above_host_guard(self):
        for floor in (2.5,4.0):
            c=copy.deepcopy(self.config);c['guard_floor_gib']=floor
            for profile in node.PROFILES:
                env=node.environment(c,c['nodes'][0],profile,'memory-check')
                self.assertGreaterEqual(float(env['TF_DS_HOST_MIN_GIB'])-floor,0.5)

    def test_plan_has_readonly_weights_and_file_auth(self):
        c=self.config;n=c['nodes'][0];command=node.docker_command(c,n,'edit','run-1')
        mounts=[command[i+1] for i,x in enumerate(command) if x=='--mount']
        self.assertIn('type=bind,src='+n['model']+',dst=/model,readonly',mounts)
        self.assertIn('type=bind,src='+n['engram']+',dst=/engram,readonly',mounts)
        self.assertIn('--api-key-file',command);self.assertNotIn('--privileged',command)
        self.assertNotIn('unless-stopped',command)
        self.assertEqual(node.environment(c,n,'edit','one')['TF_DS_SPILL_DIR'],'/cache/spill/one')
        self.assertNotEqual(node.environment(c,n,'edit','one')['TF_DS_SPILL_DIR'],node.environment(c,n,'edit','two')['TF_DS_SPILL_DIR'])

    def test_foreign_container_cannot_be_stopped(self):
        with patch.object(node,'inspect',return_value={'Config':{'Labels':{}}}),patch.object(node,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'foreign'):node.stop(self.config,self.config['nodes'][0])
            run.assert_not_called()

    def test_stop_cleans_owned_prebuild_when_no_server_exists(self):
        c=self.config;n=c['nodes'][0];name=node.prebuild_name(c,0)
        def inspect(candidate):
            return {'Config':{'Labels':{node.LABEL:c['deployment']}}} if candidate==name else None
        with patch.object(node,'inspect',side_effect=inspect),patch.object(node,'run') as run:
            self.assertTrue(node.stop(c,n)['prebuild_stopped'])
            run.assert_called_once_with(['docker','rm','-f',name])

    def test_prebuild_timeout_removes_only_its_owned_container(self):
        c=self.config;n=c['nodes'][0];name=node.prebuild_name(c,0)
        container={'Config':{'Labels':{node.LABEL:c['deployment']}}}
        def run(args,**kwargs):
            if args[:2]==['docker','run']:raise TimeoutError('compile timed out')
        with patch.object(node,'available_gib',return_value=120),patch.object(node,'inspect',side_effect=[None,container]),patch.object(node,'run',side_effect=run) as calls:
            with self.assertRaisesRegex(TimeoutError,'compile timed out'):
                node.execute({'config':c,'rank':0,'action':'prebuild'})
            self.assertEqual(calls.call_args.args[0],['docker','rm','-f',name])

    def test_snapshot_detects_original_file_changes(self):
        source=Path(self.temp.name)/'previous.conf';source.write_text('original')
        n=self.config['nodes'][0];n['previous_paths']=[str(source)]
        node.snapshot(self.config,n,'fixed')
        self.assertTrue(node.check_snapshot(self.config,n,'fixed')['ok'])
        source.write_text('changed by another operator')
        with self.assertRaisesRegex(RuntimeError,'Original files changed'):node.check_snapshot(self.config,n,'fixed')
        self.assertEqual(source.read_text(),'changed by another operator')

    def test_snapshot_detects_corrupt_backup(self):
        n=self.config['nodes'][0];node.snapshot(self.config,n,'fixed')
        (Path(n['state'])/'snapshots/fixed/configs.tar.gz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError,'backup changed'):node.check_snapshot(self.config,n,'fixed')

    def test_restoration_accepts_environment_reordering_but_rejects_changed_values(self):
        n=self.config['nodes'][0];n['previous_containers']=['original']
        old={'Name':'/original','Image':'sha256:original','State':{'Running':True},
             'Config':{'Cmd':['serve'],'Entrypoint':['engine'],'Env':['FIRST=one','SECOND=two']},'Mounts':[]}
        with patch.object(node,'inspect',return_value=old),patch.object(node,'run'):
            node.snapshot(self.config,n,'fixed')
        now=copy.deepcopy(old);now['Config']['Env']=['SECOND=two','FIRST=one']
        with patch.object(node,'inspect',return_value=now):
            self.assertTrue(node.check_snapshot(self.config,n,'fixed',containers=True)['ok'])
        now['Config']['Env']=['SECOND=changed','FIRST=one']
        with patch.object(node,'inspect',return_value=now):
            self.assertFalse(node.check_snapshot(self.config,n,'fixed',containers=True)['ok'])

    def test_restoration_preserves_duplicate_environment_key_order(self):
        n=self.config['nodes'][0];n['previous_containers']=['original']
        old={'Name':'/original','Image':'sha256:original','State':{'Running':True},
             'Config':{'Cmd':['serve'],'Entrypoint':['engine'],'Env':['A=first','B=value','A=last']},'Mounts':[]}
        with patch.object(node,'inspect',return_value=old),patch.object(node,'run'):
            node.snapshot(self.config,n,'fixed')
        now=copy.deepcopy(old);now['Config']['Env']=['B=value','A=last','A=first']
        with patch.object(node,'inspect',return_value=now):
            self.assertFalse(node.check_snapshot(self.config,n,'fixed',containers=True)['ok'])

    def test_image_mismatch_rejected_before_takeover(self):
        def both(c,action,**kw):
            if action=='status':return [{'exists':False},{'exists':False}]
            return [{'ok':True,'image_id':'a'},{'ok':True,'image_id':'b'}]
        with patch.object(cluster,'both',side_effect=both),patch.object(cluster,'snapshot') as snapshot:
            with self.assertRaisesRegex(RuntimeError,'Image IDs differ'):cluster.start(self.config,'edit',True)
            snapshot.assert_not_called()

    def test_second_rank_failure_calls_recovery(self):
        c=self.config;cluster.write_private(cluster.state_path(c),{'phase':'stopped','previous_was_active':False,'recovery_armed':True})
        def both(config,action,**kw):
            if action=='status':return [{'exists':False},{'exists':False}]
            return [{'ok':True,'image_id':'same'},{'ok':True,'image_id':'same'}]
        with patch.object(cluster,'both',side_effect=both),patch.object(cluster,'node_call',side_effect=[{'rank':0},RuntimeError('worker failed')]),patch.object(cluster,'restore') as restore:
            with self.assertRaisesRegex(RuntimeError,'worker failed'):cluster.start(c,'edit')
            restore.assert_called_once_with(c)

    def test_api_failure_does_not_disarm_recovery(self):
        c=self.config;c['previous_service']='previous.service';key=Path(self.temp.name)/'key';key.write_text('test-only-not-a-real-key')
        c['previous_health']={'key_file':str(key),'base':'http://127.0.0.1:1','model':'test'}
        cluster.write_private(cluster.state_path(c),{'phase':'running','snapshot_id':'one','previous_was_active':True,'previous_service':'previous.service'})
        with patch.object(cluster,'both',return_value=[{'ok':True},{'ok':True}]),patch.object(cluster.subprocess,'run'),patch.object(cluster,'wait_ready'),patch.object(cluster,'verify',side_effect=RuntimeError('API unavailable')),patch.object(cluster,'disarm') as disarm:
            with self.assertRaisesRegex(RuntimeError,'API unavailable'):cluster.restore(c)
            disarm.assert_not_called()
        self.assertEqual(json.loads(cluster.state_path(c).read_text())['phase'],'running')

    def test_writable_symlink_into_model_is_rejected_on_node(self):
        model=Path(self.temp.name)/'model';model.mkdir();link=Path(self.temp.name)/'alias';link.symlink_to(model,target_is_directory=True)
        c=self.config;c['nodes'][0]['model']=str(model);c['nodes'][0]['cache']=str(link)
        for action in ('start','prebuild'):
            with self.subTest(action=action),self.assertRaisesRegex(RuntimeError,'overlaps immutable'):
                node.execute({'config':c,'rank':0,'action':action})

if __name__=='__main__':unittest.main()
