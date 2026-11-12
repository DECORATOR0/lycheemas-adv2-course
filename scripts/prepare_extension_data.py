"""Prepare fixed extension samples. No model calls, result reads, or outcome-based selection.

Requires requests and pyarrow. Run from the repository root, supplying a local
full MATH500 source in the same order/ID convention as the original20 sample.
"""
from __future__ import annotations
import argparse
import io
import json
import random
from pathlib import Path
import requests
import pyarrow.parquet as pq

ARC='https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Challenge/'

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def boxed(text):
    start=text.rfind('\\boxed{')
    if start<0: raise ValueError('Missing boxed source answer')
    begin=start+7;end=begin;depth=1
    while depth and end<len(text):
        if text[end]=='{':depth+=1
        elif text[end]=='}':depth-=1
        end+=1
    if depth: raise ValueError('Unclosed boxed answer')
    return text[begin:end-1]

def math_record(index,row):
    source_id=str(row.get('id',index));reference=row['solution']
    return {'id':'math500:test:'+source_id,'benchmark':'math500','split':'test',
            'source_id':source_id,'source_index':index,'question':row.get('question',row.get('problem')),
            'answer':boxed(reference),'reference':reference,
            'metadata':{key:row[key] for key in ['unique_id','level','type'] if key in row}}

def check_unique(rows):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate selected ID')

def overlap(train,test):
    norm=lambda text:' '.join(text.casefold().split())
    shared_ids=set(r['id'] for r in train)&set(r['id'] for r in test)
    shared_questions=set(norm(r['question']) for r in train)&set(norm(r['question']) for r in test)
    if shared_ids or shared_questions:raise ValueError('Training/test overlap')
    return {'shared_ids':[],'shared_normalized_questions':0}

def arc_sample(url,split,n,seed):
    response=requests.get(url,timeout=90);response.raise_for_status()
    raw=pq.read_table(io.BytesIO(response.content)).to_pylist()
    indices=random.Random(seed).sample(range(len(raw)),n)
    selected=[]
    for index in indices:
        row=raw[index]
        selected.append({'id':f'arc_challenge_{split}_{row["id"]}',
                         'question':row.get('question',row.get('question_stem')),
                         'choices':row['choices'],'answer':row['answerKey'],
                         'source_id':row['id'],'source_index':index,'split':split,'benchmark':'arc_challenge'})
    check_unique(selected)
    return selected,{'url':url,'source_count':len(raw),'sample_count':n,'seed':seed,
                     'source_indices':indices,'selected_ids':[r['id'] for r in selected],
                     'original_option_labels_preserved':True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--math-source',required=True,help='Full500-row local JSONL source; not copied to public output')
    parser.add_argument('--base-math20',default='evidence/data/math500_test.jsonl')
    parser.add_argument('--math-train',default='evidence/data/math500_train.jsonl')
    parser.add_argument('--output',default='evidence/data/extension')
    parser.add_argument('--math-extra-seed',type=int,default=43)
    parser.add_argument('--arc-seed',type=int,default=42)
    parser.add_argument('--arc-train-url',default=ARC+'train-00000-of-00001.parquet')
    parser.add_argument('--arc-test-url',default=ARC+'test-00000-of-00001.parquet')
    args=parser.parse_args()
    raw=read_jsonl(args.math_source);base=read_jsonl(args.base_math20)
    if len(raw)!=500 or len(base)!=20:raise ValueError('Expected full500 source and original20 sample')
    all_records=[math_record(i,r) for i,r in enumerate(raw)];check_unique(all_records)
    by_id={r['id']:r for r in all_records}
    for original in base:
        if original!=by_id[original['id']]:raise ValueError('Original sample/source differs: '+original['id'])
    original_ids={r['id'] for r in base}
    remaining=[r for r in all_records if r['id'] not in original_ids]
    extra=random.Random(args.math_extra_seed).sample(remaining,80)
    math100=base+extra;check_unique(math100)
    math_overlap=overlap(read_jsonl(args.math_train),math100)
    arc_train,arc_train_meta=arc_sample(args.arc_train_url,'train',8,args.arc_seed)
    arc_test,arc_test_meta=arc_sample(args.arc_test_url,'test',100,args.arc_seed)
    arc_overlap=overlap(arc_train,arc_test)
    if {r['source_id'] for r in arc_train}&{r['source_id'] for r in arc_test}:
        raise ValueError('ARC source IDs overlap between selected train/test')
    manifest={
        'selection_basis':'Fixed source order and random seeds, without reading model outputs or scores',
        'math500':{'source_count':500,'sample_count':100,'initial_count':20,'extra_count':80,
                   'initial_20_preserved_exactly':True,'initial_source_alignment':'All fields match normalized full source',
                   'initial_seed':42,'extra_seed':args.math_extra_seed,
                   'extra_selection':'random.Random(extra_seed).sample(source-order complement of initial20,80)',
                   'initial_ids':[r['id'] for r in base],'extra_ids':[r['id'] for r in extra],
                   'initial_source_indices':[r['source_index'] for r in base],
                   'extra_source_indices':[r['source_index'] for r in extra],
                   'unique_selected_ids':len({r['id'] for r in math100}),
                   'training_test_overlap':math_overlap,
                   'public_source':'https://huggingface.co/datasets/HuggingFaceH4/MATH-500',
                   'source_order_note':'Existing research archive; original20 IDs and all fields verified against supplied full source',
                   'license':'MIT; upstream source/license references in ../SOURCES.md'},
        'arc_challenge':{'dataset':'allenai/ai2_arc','configuration':'ARC-Challenge','license':'CC-BY-SA-4.0',
                         'source_url':'https://huggingface.co/datasets/allenai/ai2_arc',
                         'train':arc_train_meta,'test':arc_test_meta,'training_test_overlap':arc_overlap,
                         'unique_selected_train_ids':len({r['id'] for r in arc_train}),
                         'unique_selected_test_ids':len({r['id'] for r in arc_test}),
                         'selection':'random.Random(seed).sample(range(source_count),n), each official split independently',
                         'training_note':'Eight candidate training examples; the run must record which four are actually used'}
    }
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    for name,rows in [('math500_test100',math100),('arc_challenge_train',arc_train),('arc_challenge_test100',arc_test)]:
        (destination/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (destination/'sampling_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'math500_test':len(math100),'arc_challenge_train_candidates':len(arc_train),
                      'arc_challenge_test':len(arc_test),'overlap_checks':'pass','original20_all_fields':'unchanged'}))

if __name__=='__main__':main()
