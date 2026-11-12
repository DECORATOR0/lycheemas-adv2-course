"""Compare matched ADv2 on/off artifacts by ID, preserving every saved result."""
import argparse
import json
from pathlib import Path
from collections import Counter

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def latest(path):return {(r:=json.loads(x))['id']:r for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}
def usage(events):
    return {'calls':len(events),'prompt_tokens':sum(x['usage'].get('prompt_tokens',0) for x in events),
            'completion_tokens':sum(x['usage'].get('completion_tokens',0) for x in events),
            'format_retries':sum(x.get('format_attempt',0)>0 for x in events)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-gate',type=Path,required=True)
    parser.add_argument('--without-gate',type=Path,required=True)
    parser.add_argument('--expected',type=int,default=100)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();on,off=args.with_gate,args.without_gate
    config_on,config_off=load(on/'config.json'),load(off/'config.json')
    differences={k:{'on':config_on.get(k),'off':config_off.get(k)} for k in set(config_on)|set(config_off) if config_on.get(k)!=config_off.get(k)}
    issues=[]
    def check(value,message):
        if not value:issues.append(message)
    check(config_on.get('evaluation_gate') is True and config_off.get('evaluation_gate') is False,'gate configuration not true/false')
    check(set(differences)=={'evaluation_gate'},'configuration has differences beyond evaluation_gate')
    selection_on,selection_off=load(on/'selection.json'),load(off/'selection.json')
    check(selection_on==selection_off,'full selected training/test records differ')
    check(load(on/'prompts.json')==load(off/'prompts.json'),'full prompts payload differs')
    check(load(on/'evaluation_prompts.json')==load(off/'evaluation_prompts.json'),'actual evaluation prompt snapshots differ')
    selected=[r['id'] for r in selection_on['test']]
    check(len(selected)==len(set(selected))==args.expected,'selected ID count differs')
    a,b=latest(on/'results.jsonl'),latest(off/'results.jsonl')
    check(set(a)==set(b)==set(selected),'result ID populations differ')
    events_on,events_off=lines(on/'calls.jsonl'),lines(off/'calls.jsonl')
    eval_on=[c for c in events_on if c['task_id'] in selected]
    eval_off=[c for c in events_off if c['task_id'] in selected]
    check(not any(c['purpose']=='adv2_judge' for c in events_off),'no-ADv2 contains judge calls')
    check(not any('\n\nVerifier feedback:\n' in c['prompt'] for c in events_off),'no-ADv2 contains verifier feedback calls')
    check(not any(c['task_id']=='optimization' for c in events_off),'no-ADv2 unexpectedly reran optimization')
    check(all(n['gate'] is None for r in b.values() for n in r.get('nodes',[])),'no-ADv2 contains gate traces')
    comparisons=[];counts=Counter({'both_correct':0,'on_correct_off_wrong':0,'on_wrong_off_correct':0,'both_wrong':0,'unfinished':0})
    for rid in selected:
        if rid not in a or rid not in b:continue
        x,y=a[rid],b[rid]
        check(x['question']==y['question'] and x['reference']==y['reference'],rid+': task/reference differs')
        done=x['status'] in ('ok','terminal_rejected') and y['status'] in ('ok','terminal_rejected')
        xc,yc=x['score']['correct'],y['score']['correct']
        category='unfinished' if not done else 'both_correct' if xc and yc else 'on_correct_off_wrong' if xc else 'on_wrong_off_correct' if yc else 'both_wrong'
        counts[category]+=1
        comparisons.append({'id':rid,'with_gate_prediction':x.get('prediction'),'without_gate_prediction':y.get('prediction'),
                            'reference':x['reference'],'with_gate_status':x['status'],'without_gate_status':y['status'],
                            'with_gate_correct':xc,'without_gate_correct':yc,'category':category})
    on_correct=sum(r['score']['correct'] for r in a.values());off_correct=sum(r['score']['correct'] for r in b.values())
    out={'with_gate_run':on.name,'without_gate_run':off.name,'expected':args.expected,
         'config_differences':differences,'validation_issues':issues,'paired_counts':dict(counts),
         'with_gate_correct':on_correct,'without_gate_correct':off_correct,
         'accuracy_difference_percentage_points':100*(on_correct-off_correct)/args.expected,
         'with_gate_evaluation_all_attempts':usage(eval_on),'without_gate_evaluation_all_attempts':usage(eval_off),
         'paired_records':comparisons,
         'interpretation':'Matched saved-score comparison; mathematical equivalence review must finish before final performance conclusions. Optimization was shared, not retrained per condition. Positive, zero, and negative differences are all retained.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:out[k] for k in ['validation_issues','paired_counts','with_gate_correct','without_gate_correct','accuracy_difference_percentage_points']}))

if __name__=='__main__':main()
