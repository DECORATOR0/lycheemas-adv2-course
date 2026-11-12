"""Real-model, controlled-message checks for pass, repair and rejection paths."""
import argparse
import asyncio
import json
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from lychee_mas.course.io import ChatClient,TASK,write_json
from lychee_mas.course.workflow import fixed_indicators
from lychee_mas.methods.runtime.agentdropout_v2 import AgentDropoutV2Controller

async def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    client=ChatClient(args.base_url,args.model,2,args.output/'calls.jsonl',1024,180)
    controller=AgentDropoutV2Controller(client.judge,max_retries=1,indicators=fixed_indicators('gsm8k'),top_k=None)
    task='A shop has four boxes. Each box holds three pencils. How many pencils are there in total?'
    good='There are 4 boxes with 3 pencils each, so 4 * 3 = 12 pencils. <answer>12</answer>'
    bad='There are 4 boxes with 3 pencils each, so add 4 + 3 = 7 pencils. <answer>7</answer>'
    rows=[]
    for label,initial,revision in [('pass',good,good),('repair',bad,good),('reject',bad,bad)]:
        token=TASK.set(label)
        async def regenerate(feedback,revision=revision): return revision
        result=await controller.process(initial,task=task,agent='controlled_probe',regenerate=regenerate,context='')
        data=result.to_dict()
        expected=(result.decision=='pass' and (label!='repair' or result.retries==1)) if label!='reject' else result.decision=='reject' and result.content is None
        rows.append({'case':label,'task':task,'initial':initial,'revision':revision,'result':data,'expected_path_observed':expected})
        TASK.reset(token)
    artifact={'type':'controlled_message_live_model_probe','benchmark_examples':False,'model':args.model,'enable_thinking':False,'passed':all(x['expected_path_observed'] for x in rows),'cases':rows,'usage':client.statistics()}
    write_json(args.output/'summary.json',artifact)
    print(json.dumps({'passed':artifact['passed'],'cases':[{k:r[k] for k in ['case','expected_path_observed']} for r in rows],'usage':artifact['usage']},indent=2))
    if not artifact['passed']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',default='Qwen3-8B');p.add_argument('--output',type=pathlib.Path,required=True)
    asyncio.run(main(p.parse_args()))
