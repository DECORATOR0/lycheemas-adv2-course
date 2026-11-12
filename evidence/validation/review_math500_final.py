"""Independently audit final append-only MATH artifacts, without changing them."""
import json
import sys
from pathlib import Path
from collections import Counter

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
sys.path.insert(0,str(REPO/'src'))
from lychee_mas.course.workflow import AGENTS,make_view,seed_prompts,tool_result
from lychee_mas.methods.prerun.maspo.executor import format_agent_prompt
from lychee_mas.course.data import score_answer
from lychee_mas.methods.prerun.maspo.textops import extract_answer

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def usage(events):
    prompt=sum(c['usage'].get('prompt_tokens',0) for c in events)
    completion=sum(c['usage'].get('completion_tokens',0) for c in events)
    return {'model_calls':len(events),'prompt_tokens':prompt,'completion_tokens':completion,
            'total_tokens':prompt+completion,
            'incomplete_outputs':sum(c.get('output_complete') is False or ('output_complete' not in c and c['finish_reason']=='length') for c in events),
            'format_retries':sum(c.get('format_attempt',0)>0 for c in events)}

p=REPO/'results/math500'
events=lines(p/'results.jsonl')
rows=list({r['id']:r for r in events}.values())
calls=lines(p/'calls.jsonl')
optim=lines(p/'optimization_calls.jsonl')
summary,selection=load(p/'summary.json'),load(p/'selection.json')
prompts=load(p/'prompts.json')['prompts']
view=make_view('math')
issues=[]
def check(ok,text):
    if not ok: issues.append(text)
check(len(rows)==20 and all(r['status']=='ok' for r in rows),'latest records not all complete')
check({r['id'] for r in rows}=={r['id'] for r in selection['test']},'selected IDs differ')
check(prompts==load(p/'evaluation_prompts.json'),'prompt snapshot differs')
check(all(prompts[a]==prompts[a+'_r2'] for a in AGENTS),'round prompts differ')
changed=[a for a in AGENTS if prompts[a]!=seed_prompts('math')[a]]
check(changed==['C']==load(p/'prompts_stats.json')['changed_roles'],'changed roles differ')
norm=lambda x:' '.join(x.casefold().split())
check(not {r['id'] for r in selection['train']} & {r['id'] for r in selection['test']},'ID overlap')
check(not {norm(r['question']) for r in selection['train']} & {norm(r['question']) for r in selection['test']},'question overlap')
check(all(c['task_id']=='optimization' for c in optim),'nonoptimization calls in copied source')
check(all(not any(r['question'] in c['prompt'] for r in selection['test']) for c in optim),'test question in optimization')
check(all(c['enable_thinking'] is False for c in calls+optim),'thinking enabled')
gates=Counter();rejects=[];transitions=[];prompt_checks=0;judge_checks=0
for row in rows:
    check([n['node'] for n in row['nodes']]==view.names,row['id']+': node order')
    accepted={}
    task_calls=[c for c in calls if c['task_id']==row['id']]
    for node in row['nodes']:
        name=node['node']
        sources=[x for x in view.predecessors[name] if accepted.get(x)]
        context='\n\n'.join(f'[{x}]\n{accepted[x]}' for x in sources)
        check(node['visible_sources']==sources,row['id']+': visible sources '+name)
        check(node['input_context']==context,row['id']+': accepted context '+name)
        role=view.specs[name].meta.get('prompt_source',name)
        expected=format_agent_prompt(prompts[role],row['question'],context)
        candidates=[c for c in task_calls if c['purpose']==name and c['prompt']==expected
                    and c.get('output_complete',c['finish_reason']!='length')]
        def returned(c):
            text=c['content'].strip()
            return tool_result(text) if name in ('C','C_r2') else text
        check(any(returned(c)==node['initial_output'] for c in candidates),row['id']+': actual initial return '+name)
        prompt_checks+=1
        gate=node['gate'];gates.update(a['decision'] for a in gate['attempts'])
        check(gate['attempts'][0]['content']==node['initial_output'],row['id']+': gate initial version '+name)
        for attempt in gate['attempts']:
            audit_matches=[]
            for call in task_calls:
                if call['purpose']!='adv2_judge' or not call.get('output_complete',call['finish_reason']!='length'):
                    continue
                original=call['prompt'].split('\n\n[Output-format retry ',1)[0]
                payload=json.loads(original.rsplit('\n',1)[1])
                if payload['agent']==name and payload['candidate']==attempt['content'] and payload['context']==context:
                    response=json.loads(call['content'])
                    audit_matches.append(response.get('feedback')==attempt['feedback'] and response.get('violations')==attempt['violations'])
            check(any(audit_matches),row['id']+': gate response has no complete logged source '+name)
        if gate['decision']=='reject':
            check(node['accepted_output'] is None and gate['content'] is None,row['id']+': rejection retained '+name)
            rejects.append({'id':row['id'],'node':name,'downstream_contexts_reconstructed':True})
        else:
            check(node['accepted_output']==gate['content']==gate['attempts'][-1]['content'],row['id']+': accepted version '+name)
        accepted[name]=node['accepted_output']
        if gate['decision']=='pass' and len(gate['attempts'])>1:
            answers=[extract_answer(a['content']) for a in gate['attempts']]
            good=[score_answer(a,row['reference'],'math',row['question'])['correct'] for a in answers]
            transitions.append({'id':row['id'],'node':name,'before':answers[0],'after':answers[-1],
                                'before_correct':good[0],'after_correct':good[-1]})
    check(row['final_output']==accepted['final'],row['id']+': final differs')
    check(score_answer(row['prediction'],row['reference'],'math',row['question'])==row['score'],row['id']+': rescored value differs')
    for c in task_calls:
        if c['purpose']=='adv2_judge':
            original=c['prompt'].split('\n\n[Output-format retry ',1)[0]
            payload=json.loads(original.rsplit('\n',1)[1])
            check(set(payload)=={'task','agent','context','candidate','indicators'},row['id']+': judge fields')
            check(payload['task']==row['question'],row['id']+': judge question')
            judge_checks+=1
check(dict(gates)==summary['gate_attempt_decisions'],'gate totals differ')
check(sum(r['score']['correct'] for r in rows)==summary['correct']==15,'correct count differs')
check(len(transitions)==summary['messages_accepted_after_retry']==6,'retry-pass count differs')
audit=load(p/'rescore_audit.json')
check([x['id'] for x in audit['changed_examples']]==['math500:test:279'],'rescore changed other items')
pirow=next(r for r in rows if r['id']=='math500:test:279')
check(pirow['original_score']==audit['changed_examples'][0]['old_score'],'old score not retained')
recorded_usage={'model_calls':sum(r['usage']['model_calls'] for r in rows),
                'prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
                'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows)}
check(all(summary[k]==v for k,v in recorded_usage.items()),'latest-record summary usage differs')
all_usage=load(p/'usage_all_attempts.json')
for label,subset in [('optimization',optim),('evaluation_all_attempts',calls),('successful_pipeline_with_resume',optim+calls)]:
    current=usage(subset)
    check(all(all_usage[label][k]==current[k] for k in current if k!='total_tokens'),'all-attempt usage differs '+label)
out={'review_date':'2026-11-12','source':'results/math500','append_only_result_events':len(events),
     'latest_completed_examples':len(rows),'correct':15,'errors':0,'prompt_application_checks':prompt_checks,
     'judge_payload_checks':judge_checks,'changed_roles':changed,'gate_attempt_decisions':dict(gates),
     'rejection_checks':rejects,'accepted_after_retry_review':transitions,
     'usage_optimization_counted_once':usage(optim),'usage_evaluation_all_attempts':usage(calls),
     'usage_pipeline_with_resume':usage(optim+calls),'usage_latest_completed_records':recorded_usage,
     'rescore_audit':audit,'required_fixes':issues,
     'manual_recovered_items':[
       {'id':'math500:test:327','prediction':'10','reference':'15','correct':False,
        'reason':'Writing P=(x-1)(x-4)(x-c)+ax+b gives 2a+b=2c-4 and 3a+b=2c-6, hence a=-2,b=-5,c=-5/2 and P(5)=15. The response drops b while simplifying the first equation.'},
       {'id':'math500:test:377','prediction':'6','reference':'6','correct':True,
        'reason':'4!=24; the exponent tower is even, and powers of4 end in6 for every even exponent.'}],
     'interpretation_notes':['The 22 append-only records retain two earlier error events; the latest record per ID yields20 completed examples.',
       'Incomplete client outputs are retained and charged but not returned to the graph; matching uses complete generated content and the actual prompt, not the first same-purpose event.',
       'The six retry-pass messages contain one correct-to-wrong, two correct-to-correct, and three wrong-to-wrong transitions.',
       'The copied790-call optimization source is counted once. Do not add its parent artifact total again.',
       'summary.json costs sum the latest successful record usage; full evaluation costs include earlier failed attempts and are given in usage_all_attempts.json.']}
(HERE/'math500_live_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:out[k] for k in ['latest_completed_examples','correct','prompt_application_checks','required_fixes','usage_pipeline_with_resume']},ensure_ascii=False))
