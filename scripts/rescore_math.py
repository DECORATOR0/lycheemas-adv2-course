"""Recompute all completed MATH answers; preserve prior scores and record every change."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from lychee_mas.course.data import score_answer

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=pathlib.Path);p.add_argument('--write',action='store_true');args=p.parse_args()
 rows=[json.loads(x) for x in (args.run/'results.jsonl').read_text(encoding='utf8').splitlines()]
 latest={r['id']:r for r in rows};changes=[];checked=0
 for row in latest.values():
  if row['status']!='ok':continue
  checked+=1;score=score_answer(row['prediction'],row['reference'],'math',row['question'])
  if score['correct']!=row['score']['correct']:
   changes.append({'id':row['id'],'prediction':row['prediction'],'reference':row['reference'],'old_score':row['score'],'new_score':score})
   row.setdefault('original_score',row['score']);row['score']=score
   row['scoring_revision']='unicode_pi_normalization'
 artifact={'checked_examples':checked,'changed_examples':changes,'correct_after':sum(bool(r['score']['correct']) for r in latest.values()),'reason':'Uniform deterministic answer equivalence scoring; Unicode pi and LaTeX pi normalized identically. Model outputs are unchanged.'}
 if args.write:
  (args.run/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
  summary=json.loads((args.run/'summary.json').read_text(encoding='utf8'))
  summary['correct']=artifact['correct_after'];summary['accuracy_full_denominator']=artifact['correct_after']/summary['selected_examples'];summary['scoring_review']='rescore_audit.json'
  (args.run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
  audit=args.run/'rescore_audit.json'
  if changes or not audit.exists():audit.write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(artifact,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
