"""Read-only structural and scoring-replay QA for expanded benchmark artifacts.

Writes a separate review, never modifies source scores or reruns inference.
Mathematical representation cases and incorrect answers require manual review;
automatic answer-match transitions are not automatically successful corrections.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent.parent
sys.path.insert(0,str(REPO/'src'))
from lychee_mas.course.workflow import AGENTS,make_view,tool_result,seed_prompts
from lychee_mas.methods.prerun.maspo.executor import format_agent_prompt
from lychee_mas.methods.prerun.maspo.textops import extract_answer
from lychee_mas.course.data import score_answer

def load(path): return json.loads(path.read_text(encoding='utf-8'))
def lines(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def latest(events): return list({r['id']:r for r in events}.values())
def base_prompt(call): return call['prompt'].split('\n\n[Output-format retry ',1)[0]
def valid_judge_schema(call):
    if call.get('purpose')!='adv2_judge':return True
    try:
        value=json.loads(call['content'])
        return (isinstance(value,dict) and set(value)=={'feedback','violations'} and
                isinstance(value['feedback'],str) and isinstance(value['violations'],list) and
                all(isinstance(x,str) for x in value['violations']))
    except (ValueError,TypeError):return False
def complete(call):
    return call.get('output_complete',call.get('finish_reason')!='length') and valid_judge_schema(call)
def usage(calls):
    prompt=sum(c['usage'].get('prompt_tokens',0) for c in calls)
    completion=sum(c['usage'].get('completion_tokens',0) for c in calls)
    return {'calls':len(calls),'prompt_tokens':prompt,'completion_tokens':completion,'total_tokens':prompt+completion,
            'incomplete_outputs':sum(not complete(c) for c in calls),
            'invalid_judge_schema_outputs':sum(not valid_judge_schema(c) for c in calls),
            'format_retries':sum(c.get('format_attempt',0)>0 for c in calls)}
def coverage(selected):
    subjects=Counter();levels=Counter();cross=Counter()
    for row in selected:
        meta=row.get('metadata',{})
        unique=meta.get('unique_id','').split('/')
        subject=meta.get('type',meta.get('subject',unique[1] if len(unique)>2 else 'unknown'))
        level=str(meta.get('level','unknown'))
        subjects[subject]+=1;levels[level]+=1;cross[subject+' / '+level]+=1
    return {'subject_from_type_subject_or_unique_id':dict(subjects),'level':dict(levels),'subject_level':dict(cross)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--expected',type=int,default=100)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--manifest',type=Path,default=REPO/'evidence/data/extension/sampling_manifest.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run
    config,selection=load(run/'config.json'),load(run/'selection.json')
    summary=load(run/'summary.json') if (run/'summary.json').exists() else None
    prompts=load(run/'prompts.json')['prompts']
    events=lines(run/'results.jsonl') if (run/'results.jsonl').exists() else []
    rows=latest(events);byid={r['id']:r for r in rows}
    calls=lines(run/'calls.jsonl')
    optimization=[c for c in calls if c['task_id']=='optimization']
    copied_optimization=lines(run/'optimization_calls.jsonl') if (run/'optimization_calls.jsonl').exists() else []
    evaluation=[c for c in calls if c['task_id']!='optimization']
    benchmark=config['benchmark'];view=make_view(benchmark)
    selected={r['id']:r for r in selection['test']}
    issues=[]
    def check(condition,description):
        if not condition:issues.append(description)
    check(len(selected)==len(selection['test'])==args.expected,'selection count or unique IDs differ')
    check(set(byid)<=set(selected),'unexpected result IDs')
    check(prompts==load(run/'evaluation_prompts.json'),'evaluation prompt snapshot differs')
    check(all(prompts[a]==prompts[a+'_r2'] for a in AGENTS),'shared round prompts differ')
    check(not optimization or not copied_optimization,'both inline and copied optimization present: do not double count')
    selected_optim=optimization or copied_optimization
    all_calls=evaluation+selected_optim
    check(all(c['enable_thinking'] is False for c in all_calls),'thinking enabled')
    norm=lambda s:' '.join(s.casefold().split())
    check(not {r['id'] for r in selection['train']}&set(selected),'training/test IDs overlap')
    check(not {norm(r['question']) for r in selection['train']}&{norm(r['question']) for r in selected.values()},'training/test questions overlap')
    check(all(not any(r['question'] in c['prompt'] for r in selected.values()) for c in selected_optim),'test question in optimization')
    grouped=defaultdict(list)
    for call in evaluation:grouped[call['task_id']].append(call)
    gate_counts=Counter({'pass':0,'retry':0,'reject':0});rejected=[];transitions=[];candidates=[];prompt_checks=0;audit_checks=0
    if not config.get('evaluation_gate',True):
        check(not any(c['purpose']=='adv2_judge' for c in evaluation),'disabled gate still made judge calls')
        check(not any('\n\nVerifier feedback:\n' in c['prompt'] for c in evaluation),'disabled gate still made feedback calls')
    for row in rows:
        if row['status'] not in ('ok','terminal_rejected'):continue
        rid=row['id'];question=selected[rid]['question'];task_calls=grouped[rid]
        check(row['question']==question and row['reference']==selected[rid]['answer'],rid+': task/reference differs')
        check([n['node'] for n in row['nodes']]==view.names,rid+': node order differs')
        accepted={};judges=[]
        for call in task_calls:
            if call['purpose']=='adv2_judge':
                payload=json.loads(base_prompt(call).rsplit('\n',1)[1])
                check(set(payload)=={'task','agent','context','candidate','indicators'},rid+': judge fields differ')
                check(payload['task']==question,rid+': judge task differs')
                if complete(call):judges.append((payload,json.loads(call['content'])))
        for node in row['nodes']:
            name=node['node'];sources=[p for p in view.predecessors[name] if accepted.get(p)]
            context='\n\n'.join(f'[{p}]\n{accepted[p]}' for p in sources)
            check(node['visible_sources']==sources,rid+': visible sources '+name)
            check(node['input_context']==context,rid+': accepted-only context '+name)
            role=view.specs[name].meta.get('prompt_source',name)
            expected=format_agent_prompt(prompts[role],question,context)
            generation=[c for c in task_calls if c['purpose']==name and complete(c)]
            def returned(call):
                text=call['content'].strip()
                return tool_result(text) if name in ('C','C_r2') and benchmark in ('math','gsm8k') else text
            check(any(base_prompt(c)==expected and returned(c)==node['initial_output'] for c in generation),rid+': complete initial call '+name)
            prompt_checks+=1;gate=node['gate']
            if gate is None:
                check(not config.get('evaluation_gate',True),rid+': enabled gate missing '+name)
                check(node['accepted_output']==node['initial_output'],rid+': ungated output changed '+name)
                accepted[name]=node['accepted_output']
                continue
            check(config.get('evaluation_gate',True),rid+': disabled gate has audit '+name)
            gate_counts.update(a['decision'] for a in gate['attempts'])
            check(node['initial_output']==gate['attempts'][0]['content'],rid+': initial gate content '+name)
            for i,attempt in enumerate(gate['attempts']):
                check(any(p['agent']==name and p['context']==context and p['candidate']==attempt['content'] and
                          verdict.get('feedback')==attempt['feedback'] and verdict.get('violations')==attempt['violations']
                          for p,verdict in judges),rid+': complete judge source '+name+':'+str(i))
                check(any(returned(c)==attempt['content'] and
                          (base_prompt(c)==expected or base_prompt(c).startswith(expected+'\n\nVerifier feedback:\n'))
                          for c in generation),rid+': complete generation source '+name+':'+str(i))
                audit_checks+=1
            if gate['decision']=='reject':
                check(node['accepted_output'] is None and gate['content'] is None,rid+': rejected content retained '+name)
                rejected.append({'id':rid,'node':name,'accepted_output_absent':True})
            else:check(node['accepted_output']==gate['content']==gate['attempts'][-1]['content'],rid+': accepted version '+name)
            accepted[name]=node['accepted_output']
            if gate['decision']=='pass' and len(gate['attempts'])>1:
                answers=[extract_answer(a['content']) for a in gate['attempts']]
                matching=[score_answer(a,row['reference'],benchmark,question)['correct'] for a in answers]
                transitions.append({'id':rid,'node':name,'reference':row['reference'],'before':answers[0],'after':answers[-1],
                                    'before_label_match':matching[0],'after_label_match':matching[-1],
                                    'classification_requires_manual_review':True})
        check((row['final_output'] or None)==accepted['final'],rid+': final output differs')
        replay=score_answer(row['prediction'],row['reference'],benchmark,question)
        check(replay['correct']==row['score']['correct'],rid+': score replay differs')
        representation=bool(benchmark=='math' and re.search(r',|_|\\begin|\\text|\\circ|<|>|\\infty|\\sqrt\[|\\pm',row['reference']))
        if not row['score']['correct'] or representation:
            candidates.append({'id':rid,'question':question,'prediction':row['prediction'],'reference':row['reference'],
                               'score':row['score'],'representation_check':representation})
    completed=[r for r in rows if r['status'] in ('ok','terminal_rejected')];errors=[r['id'] for r in rows if r['status']=='error']
    finished=len(rows)==args.expected and not errors
    correct=sum(r['score']['correct'] for r in completed)
    if summary and summary.get('recorded_examples')==len(rows):
        check(summary['correct']==correct,'summary correct count')
        check(summary['gate_attempt_decisions']==dict(gate_counts),'summary gate totals')
        check(summary['messages_accepted_after_retry']==len(transitions),'summary retry-pass count')
    out={'run_name':run.name,'state':'structural_review_complete_manual_scoring_pending' if finished else 'partial',
         'selected':len(selected),'append_only_result_events':len(events),'latest_records':len(rows),
         'completed':len(completed),'error_ids':errors,'missing_ids':sorted(set(selected)-set(byid)),
         'terminal_rejected':sum(r['status']=='terminal_rejected' for r in rows),
         'correct_saved':correct,'accuracy_saved_full_denominator':correct/len(selected),
         'changed_roles':[a for a in AGENTS if prompts[a]!=seed_prompts(benchmark)[a]],
         'prompt_application_checks':prompt_checks,'complete_audit_source_checks':audit_checks,
         'gate_attempt_decisions':dict(gate_counts),'rejection_checks':rejected,
         'retry_pass_transitions':transitions,'manual_scoring_candidates':candidates,
         'coverage_all_selected':coverage(list(selected.values())),
         'usage_evaluation_all_attempts':usage(evaluation),'usage_optimization_counted_once':usage(selected_optim),
         'usage_artifact_total':usage(all_calls),'structural_or_score_replay_issues':issues,
         'cost_note':'Frozen previously optimized prompts do not incur new optimization; a copied source optimization is counted once only.'}
    if args.baseline:
        old=latest(lines(args.baseline/'results.jsonl'));old_byid={r['id']:r for r in old}
        manifest=load(args.manifest)['math500'];first=set(manifest['initial_ids']);extra=set(manifest['extra_ids'])
        check(set(selected)==first|extra and not first&extra,'extension100 membership differs from frozen manifest')
        out['initial20_comparison']=[{'id':rid,'old_prediction':old_byid[rid]['prediction'],
                                     'old_correct':old_byid[rid]['score']['correct'],
                                     'new_prediction':byid[rid].get('prediction'),'new_status':byid[rid]['status'],
                                     'new_correct':byid[rid]['score']['correct']}
                                    for rid in manifest['initial_ids'] if rid in byid]
        for label,ids in [('initial20',first),('new80',extra)]:
            subset=[r for r in completed if r['id'] in ids]
            out[label]={'selected':len(ids),'completed':len(subset),'correct_saved':sum(r['score']['correct'] for r in subset),
                        'coverage':coverage([selected[rid] for rid in ids])}
        out['participant_prompts_equal_baseline']=prompts==load(args.baseline/'prompts.json')['prompts']
        old_prefix={base_prompt(c).rsplit('\n',1)[0] for c in lines(args.baseline/'calls.jsonl') if c['purpose']=='adv2_judge'}
        new_prefix={base_prompt(c).rsplit('\n',1)[0] for c in evaluation if c['purpose']=='adv2_judge'}
        out['judge_prompt_prefix_equal_baseline']=new_prefix==old_prefix
    manual_path=HERE/'math_scoring_manual_review.json'
    if benchmark=='math' and manual_path.exists():
        manual=load(manual_path).get('conditions',{}).get(run.name)
        if manual:
            check(correct==manual['correct_after_uniform_review'],'manual-reviewed correct count differs')
            for item in manual['incorrect_answer_checks']:
                current=byid[item['id']]
                check(current.get('prediction')==item['prediction'] and current['reference']==item['reference'],item['id']+': manual review answer changed')
                check(current['score']['correct']==(item['decision']=='equivalent_representation'),item['id']+': manual reviewed score differs')
            for item in manual['correct_nonidentical_expression_checks']:
                current=byid[item['id']]
                check(current['prediction']==item['prediction'] and current['reference']==item['reference'] and current['score']['correct'],item['id']+': manually verified positive changed')
            out['manual_review_source']=manual_path.name
            if finished and not issues:out['state']='review_complete'
    elif benchmark in ('arc_easy','arc_challenge','openbookqa'):
        exact=all((r.get('prediction','').strip()==r['reference'].strip())==r['score']['correct'] for r in completed)
        out['independent_exact_label_checks']=len(completed) if exact else 0
        if exact and finished and not issues:out['state']='review_complete'
    target=args.output or HERE/(run.name+'_review.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:out[key] for key in ['state','selected','completed','correct_saved','prompt_application_checks','structural_or_score_replay_issues']},ensure_ascii=False))

if __name__=='__main__':main()
