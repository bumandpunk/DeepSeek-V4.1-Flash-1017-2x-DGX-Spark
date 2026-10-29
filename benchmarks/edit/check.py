"""Independent bounded edit fixtures; executed only inside the code sandbox."""
import importlib.util,random,sys
name,path=sys.argv[1:]
spec=importlib.util.spec_from_file_location('submission',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
n=0
def check(ok):
 global n
 n+=1
 assert ok, 'assertion '+str(n)
if name=='bisect':
 rng=random.Random(1017)
 check('bisect_bounds' in m.__all__ if hasattr(m,'__all__') else True)
 for _ in range(200):
  values=sorted(rng.randrange(12) for _ in range(rng.randrange(40))); x=rng.randrange(12)
  lo=rng.randrange(len(values)+1); hi=rng.randrange(lo,len(values)+1)
  check(m.bisect_bounds(values,x,lo,hi)==(m.bisect_left(values,x,lo,hi),m.bisect_right(values,x,lo,hi)))
 check(m.bisect_bounds([],5)==(0,0))
 check(m.bisect_bounds([{'v':1},{'v':3},{'v':3},{'v':7}],3,key=lambda x:x['v'])==(1,3))
 a=[1,4,4,8];m.insort_left(a,4);check(a==[1,4,4,4,8])
elif name=='graphlib':
 T=m.TopologicalSorter
 check(len(T())==0)
 t=T({'c':['a','b'],'d':['c']});check(len(t)==4);t.prepare()
 check(set(t.get_ready())=={'a','b'});t.done('a','b');check(t.get_ready()==('c',));t.done('c');check(t.get_ready()==('d',));t.done('d');check(not t.is_active());check(len(t)==4)
 for size in range(1,60):
  deps={i:list(range(i)) for i in range(size)};t=T(deps);check(len(t)==size);check(list(t.static_order())==list(range(size)))
 try:T({'a':['b'],'b':['a']}).prepare()
 except m.CycleError:check(True)
 else:raise AssertionError('cycle no longer detected')
elif name=='textwrap':
 check('wrap_lines' in m.__all__)
 fixtures=[('',[]),('\n',['']),('alpha beta\n\ngamma',['alpha','beta','','gamma']),('   \nword',['','word']),('ab cd\r\nef gh',['ab cd','ef gh'])]
 for source,want in fixtures:check(m.wrap_lines(source,width=5)==want)
 check(m.wrap_lines('hello world\na b',width=8,initial_indent='> ')==['> hello','world','> a b'])
 for size in range(1,80):
  text='one two three four five\nsix seven eight nine\n\nend'
  want=[]
  for line in text.splitlines():want.extend(m.wrap(line,width=size) or [''])
  check(m.wrap_lines(text,width=size)==want)
 check(m.dedent('  a\n  b')=='a\nb');check(m.fill('one two three',width=7)=='one two\nthree')
else:raise ValueError(name)
print('{"passed":%d,"total":%d}'%(n,n))
