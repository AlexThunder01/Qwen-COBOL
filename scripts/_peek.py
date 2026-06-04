import json
d = json.load(open('results/sft_step300_qwen36_27b.json'))
for r in d['results'][:3]:
    resp = r['response']
    fence = '```cobol' in resp.lower()
    ident = 'IDENTIFICATION' in resp.upper()
    print('='*60)
    print(f"TASK: {r['task_id']} | compiles: {r['compiles']} | len: {len(resp)} char")
    print(f"  contiene ```cobol: {fence} | IDENTIFICATION: {ident}")
    print('--- ULTIMI 600 char ---')
    print(resp[-600:])
    print()
