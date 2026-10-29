"""Render the recorded second-round comparison as a standalone HTML report."""
import argparse, html, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAMES={'e320-base-v2':'E320 · 前置基线','e320-final-v1':'E320 · 融合内核',
       'e320-return-baseline':'E320 · 回切基线','a0-text-baseline':'A0 · Python 基线',
       'a0-text-fused':'A0 · 融合内核','a0-mia-native':'A0 · Mia 原生','a0-return-decode':'A0 · 回切短输出'}

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--source-zip',type=Path);a=p.parse_args()
    s=json.loads((ROOT/'results/2026-10-28/summary.json').read_text())
    q=s['quality']; arms=s['speed']; final=arms['e320-final-v1']; baseline=arms['e320-return-baseline']
    gain=(final['prefill']['131072']['tps_median']/baseline['prefill']['131072']['tps_median']-1)*100
    code=(final['decode']['code']['decode_tps_median']/baseline['decode']['code']['decode_tps_median']-1)*100
    sc=q['scicode']; can=q['canitedit']
    returned=arms.get('a0-return-decode')
    return_note=''
    if returned:
        base_code=returned['decode']['code']['decode_tps_median'];candidate_code=arms['a0-text-fused']['decode']['code']['decode_tps_median']
        return_note=f'A0 首轮代码生成基线为 92.49 tok/s，回切后为 {base_code:.2f}；融合版本为 {candidate_code:.2f}。对回切基线仅 {100*(candidate_code/base_code-1):+.2f}%，初始约 2.8% 的表面收益不作为稳定结论。'
    selected={k:v for k,v in arms.items() if k in NAMES}
    payload=json.dumps({'arms':selected,'names':NAMES},ensure_ascii=False).replace('<','\\u003c')
    def rows(labels):
        out=[]
        for label in labels:
            r=arms.get(label)
            if not r:continue
            values=[]
            for kind in ('code','prose','structured'):
                v=r['decode'].get(kind,{}).get('decode_tps_median')
                values.append(f'{v:.2f}' if v is not None else '未完成')
            for length in ('8192','32768','131072'):
                v=r['prefill'].get(length,{}).get('tps_median')
                values.append(f'{v:,.1f}' if v is not None else ('未测' if r['manifest'].get('profile_only') else '未完成'))
            out.append('<tr><th scope="row">'+html.escape(NAMES[label])+'</th>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
        return ''.join(out)
    source_links='''<a href="https://github.com/bertholomus/TensorFold">TensorFold</a> ·
    <a href="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks">MiaAI-Lab</a> ·
    <a href="https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark">Elevatormusic</a> ·
    <a href="https://github.com/lucataco/DeepSeek-V4.1-Flash-2x-DGX-Spark">lucataco</a> ·
    <a href="https://github.com/sudoingX/dgx-spark-deepseek-v41">sudoingX</a>'''
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spark1017 · 普通生成与长上下文实验</title>
<style>
:root{color-scheme:light;--ink:#172429;--muted:#52656c;--line:#d7e0df;--accent:#116459;--paper:#f5f7f4}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:auto;padding:52px 26px 70px}a{color:#005d96;text-underline-offset:3px}h1{font-size:clamp(30px,4.5vw,48px);line-height:1.2;letter-spacing:-.04em;margin:16px 0 22px}h2{font-size:25px;margin:44px 0 16px}h3{font-size:18px;margin:20px 0 8px}p{max-width:900px;margin:12px 0}.eyebrow{letter-spacing:.12em;font-size:12px;font-weight:700;color:var(--accent)}.lead{font-size:20px;max-width:850px}.muted,small{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:30px 0}.card{border-top:3px solid var(--accent);background:white;padding:22px;border-radius:0 0 8px 8px}.value{display:block;font-size:34px;line-height:1.3;font-weight:700;font-variant-numeric:tabular-nums}.card small{display:block;margin-top:9px}.panel{background:white;padding:24px;border:1px solid var(--line);border-radius:10px;margin:18px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:13px 11px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child{text-align:left}thead th{color:var(--muted);font-size:12px}caption{text-align:left;font-weight:650;margin:0 0 10px;font-size:17px}select{font:inherit;padding:7px 10px;min-height:44px;background:white;border:1px solid #829b99;border-radius:5px;margin:6px 16px 18px 8px}.barrow{display:grid;grid-template-columns:190px 1fr 105px;align-items:center;gap:14px;margin:16px 0;font-size:14px}.track{height:20px;background:#ecf0ed;border-radius:3px}.bar{height:100%;background:var(--accent);border-radius:3px}.barrow span:last-child{text-align:right;font-variant-numeric:tabular-nums}.path{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:20px 0}.path span{padding:9px 13px;border:1px solid var(--line);background:#f7faf8;border-radius:4px}li{margin:9px 0}code{font-size:13px;background:#eaf0ed;padding:2px 5px;border-radius:3px}details{margin:22px 0}summary{cursor:pointer;font-weight:650}footer{border-top:1px solid var(--line);margin-top:45px;padding-top:22px;font-size:13px;color:var(--muted)}@media(max-width:740px){main{padding:28px 16px}.cards{grid-template-columns:1fr}.card{padding:17px}.value{font-size:30px}.panel{padding:16px}.barrow{grid-template-columns:140px 1fr 75px;gap:7px;font-size:12px}h1{font-size:34px}}
</style><main>
<div class="eyebrow">SPARK1017 / ENGINE EXPERIMENT 02 / 2026.10.28</div>
<h1>普通生成与长上下文：<br>通信融合的收益和边界</h1>
<p class="lead">新增分段 RDMA 接收与 HC 后处理融合。长上下文预填充取得小幅收益；普通生成仍没有稳定提速，也未达到“通用场景超过开源方案”的目标。</p>
<p class="muted">两台 DGX Spark / GB10 推理，RTX4090 主机运行公共测量客户端和隔离评分。以下数值由附带的完整响应记录自动生成。</p>
<div class="cards"><div class="card">128K 预填充<span class="value">__GAIN__</span><small>E320 融合内核相对回切基线；3 次中位数</small></div><div class="card">普通代码生成<span class="value">__CODE__</span><small>9 次中位数；仍按未证实稳定提速处理</small></div><div class="card">SciCode 开发回归<span class="value">__SC__ / 50</span><small>与基线相同；50 个代码与逐题判定一致</small></div></div>
<h2>相同权重，分别比较</h2>
<p>E320 与 A0 是两个独立对照组。E320 删除了部分专家，模型函数已改变；它与 A0 的差异不计作引擎提速。A0 三个引擎使用同一套现有 EXL3 2.9 bpw 权重字节。</p>
<div class="panel scroll"><table><caption>E320：新内核与前、后基线</caption><thead><tr><th>引擎</th><th>代码 tok/s</th><th>散文 tok/s</th><th>结构化 tok/s</th><th>8K 预填充</th><th>32K 预填充</th><th>128K 预填充</th></tr></thead><tbody>__E320_ROWS__</tbody></table></div>
<div class="panel scroll"><table><caption>A0：同条件开源引擎对照</caption><thead><tr><th>引擎</th><th>代码 tok/s</th><th>散文 tok/s</th><th>结构化 tok/s</th><th>8K 预填充</th><th>32K 预填充</th><th>128K 预填充</th></tr></thead><tbody>__A0_ROWS__</tbody></table></div>
<p>__RETURN_NOTE__</p><p class="muted">预填充单位均为输入 tok/s，含首 token 等待时间。回切短输出一行仅复测普通生成，长输入未重复。若某项未完成，保留失败记录，不用推算数填表。短输出吞吐受流式合包影响，原始记录同时保留 TTFT 与整次耗时。</p>
<div class="panel"><label for="group">权重组</label><select id="group"><option value="e320">E320</option><option value="a0">A0</option></select><label for="metric">指标</label><select id="metric"><option value="long">128K 预填充</option><option value="code">普通代码生成</option><option value="prose">散文生成</option><option value="structured">结构化生成</option></select><div id="bars" aria-live="polite"></div><small>条形图从零起算；未完成项没有条形。</small></div>
<h2>本轮具体改了什么</h2>
<div class="panel"><div class="path"><span>两段数据入队</span>→<span>双网口 RDMA 传输</span>→<span>按段读取接收环</span>→<span>直接更新 HC 残差</span></div><p>保留上游注册、代理、完成通知和异常中止机制。新增 GPU 序号 ticket、双段流水线及直接消费接收环的 CUDA 内核，省去完整 gathered 中间张量的写入和读取。注册环约 51 MiB。</p><p>加法仍按 rank 0、rank 1 顺序，保留 FP32 运算、BF16 舍入边界与 HC 的 FMA 顺序。解码侧复用上游数值内核与侧流调度。权重、采样和主模型验收规则均未更改。</p><p><strong>局部收益不能当作整机收益：</strong>2048 行微基准中，通信加 HC 后处理从 3.56 ms 降至 2.59 ms，约减少 27%；端到端长上下文仅取得上表中的小幅收益。</p></div>
<h2>质量与正确性</h2><ul>
<li>最终 E320 速度套件 58/58 配对完整文本和 token 数相同；回切基线的结果另行核对。</li>
<li>SciCode 固定 50 子题：__SC__ / 50，完整父题 __PARENTS__ / 15。50 个请求、抽取代码和逐题判定均与冻结基线一致。</li>
<li>CanItEdit 固定 12 题、两次重复：共 __CAN__ / 24，全部配对文本、代码和判定一致。失败题完整保留。</li>
<li>双机 eager / CUDA Graph 多形状逐位一致；100 个改变载荷并错开两端时序的事务通过；HC 侧流有效状态与图重放校验通过。</li>
</ul><p class="muted">这些是开发回归与推理一致性证据。SciCode 的参考答案控制此前仅通过 48/50；本报告不是 Artificial Analysis 官方成绩，也未证明超过原始模型官方分数。</p>
<h2>没有采用的路线</h2><div class="panel scroll"><table><thead><tr><th>实现</th><th>实测结论</th><th>本轮选择</th></tr></thead><tbody>
<tr><th>4090 增量同步 DSpark</th><td>RPC 约 11.5 ms；筛选生成慢约 18–21%</td><td>关闭</td></tr><tr><th>专家 occupancy 工作列表</th><td>小批量部分获益，2048 行无稳定收益</td><td>关闭</td></tr><tr><th>持久化专家任务队列</th><td>逐位一致，代表负载慢约 1–10%</td><td>关闭</td></tr><tr><th>直接 fragment 链</th><td>逐位一致，多数负载变慢</td><td>关闭</td></tr><tr><th>控制消息合并</th><td>完整套件变化约 0–1%</td><td>关闭</td></tr><tr><th>预填充 HC 系数侧流</th><td>逐位一致，端到端收益回退</td><td>关闭</td></tr></tbody></table></div>
<h2>怎样复核</h2><p>短输出 384 tokens，每类 9 次；8K / 32K / 128K 冷输入各 3 次；另含并发 2/4 和 128K 后 512-token 解码。容量 524288、两槽、2048-token chunk、五个 DSpark 草稿。A0 为纯文本对照；E320 保留视觉塔。</p><p>保留上游 encoder prefill 加有界 decoder replay，不声称是完整精确 decoder prefill。每次重启用新 spill 目录，冷输入日志应为 cached=0。未改变功耗或频率，保留主机内存保护；顺序测试与少量重复仍限制了小幅差异的统计结论。</p>
<p><a href="Spark1017/results/2026-10-28/summary.json">机器可读结果</a> · <a href="Spark1017/results/2026-10-28/README.md">完整结果与测量边界</a> · <a href="Spark1017/README.md">源码说明</a> · <a href="Spark1017/docs/round2-design.md">引擎设计</a> · <a href="Spark1017/results/2026-10-28/cold-prefix-validation.json">冷前缀核验</a> · <a href="Spark1017/results/2026-10-28/deployment-restore.json">原部署恢复验证</a>__SOURCE_ZIP_LINK__</p>
<details><summary>固定来源与复现边界</summary><p>TensorFold 基线固定在 <code>07aa02585f64</code>，附原部署安全补丁。Mia 固定在 <code>27741e176a9e</code>，使用其公开补丁构建原生程序。更早的 lucataco 内存失败与 vLLM 格式不兼容记录保留，不能据此宣称全面排名第一。</p><p>已封存 E320 实测源码。发布版的许可证注释与版本号差异单独披露；五个引擎覆盖文件可由固定上游重建。实际验证使用保留镜像加源码挂载；Dockerfile 的全新镜像构建不冒充已验证结果。</p><p>复核记录可运行 <code>python3 tools/summarize_round2.py</code> 与 <code>python3 tools/check_round2.py</code>。这些命令不重新执行模型，也不运行生成的代码。</p></details>
<footer><p>相关开源工作：__SOURCES__</p><p>新增代码以 Apache-2.0 提供，并保留上游许可证。项目未训练新的基础模型，不分发权重、密钥、私有配置或 rank cache。分段传输、计算通信融合和推测解码都有先例；这里的贡献是具体实现及其可复核实机结果。</p></footer>
</main><script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),g=document.getElementById('group'),m=document.getElementById('metric'),area=document.getElementById('bars');
function draw(){let entries=Object.entries(data.arms).filter(([k])=>k.startsWith(g.value)).map(([k,v])=>({label:data.names[k],missing:v.manifest.profile_only?'未测':'未完成',value:m.value==='long'?v.prefill['131072']?.tps_median:v.decode[m.value]?.decode_tps_median}));const max=Math.max(1,...entries.map(x=>x.value||0));area.replaceChildren();for(const x of entries){const row=document.createElement('div');row.className='barrow';const label=document.createElement('span');label.textContent=x.label;const track=document.createElement('div');track.className='track';const bar=document.createElement('div');bar.className='bar';bar.style.width=(100*(x.value||0)/max)+'%';track.append(bar);const val=document.createElement('span');val.textContent=x.value===undefined?x.missing:x.value.toFixed(1)+' tok/s';row.append(label,track,val);area.append(row);}}
g.addEventListener('change',draw);m.addEventListener('change',draw);draw();</script></html>'''
    mapping={'__GAIN__':f'{gain:+.2f}%','__CODE__':f'{code:+.2f}%','__SC__':str(sc['candidate']['passed']),
        '__PARENTS__':str(sc['candidate']['parent_all_passed']),'__CAN__':str(can['candidate_passed']),
        '__E320_ROWS__':rows(['e320-base-v2','e320-final-v1','e320-return-baseline']),
        '__A0_ROWS__':rows(['a0-text-baseline','a0-text-fused','a0-mia-native','a0-return-decode']),
        '__RETURN_NOTE__':return_note,'__SOURCES__':source_links,'__DATA__':payload}
    for key,value in mapping.items():document=document.replace(key,value)
    prefix=Path(os.path.relpath(ROOT,a.output.resolve().parent)).as_posix()
    document=document.replace('href="Spark1017/', 'href="'+html.escape(prefix,quote=True)+'/')
    zip_link=''
    if a.source_zip:
        href=Path(os.path.relpath(a.source_zip.resolve(),a.output.resolve().parent)).as_posix()
        zip_link=' · <a href="'+html.escape(href,quote=True)+'">完整源码与证据</a>'
    document=document.replace('__SOURCE_ZIP_LINK__',zip_link)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(document)
    print(str(a.output))

if __name__=='__main__':main()
