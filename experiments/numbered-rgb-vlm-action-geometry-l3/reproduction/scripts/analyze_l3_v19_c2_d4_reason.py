#!/usr/bin/env python3
"""Detailed reason analysis for v19 condition C2 D4."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/vlm_action_geometry_single_info_l3_primary_blocker_v19"
OUT = EXP / "results/c2_d4_reason"
VIS = OUT / "visualizations"
REASONS = ["OCCLUSION", "CLEARANCE_OVERLAP", "BOTH", "NONE"]
FAMILIES = ["FC_CLEAR", "FC_BLOCKED", "TRANSLATE", "ROTATE", "LIFT_AND_RELOCATE"]
SEEDS = [28101, 28102, 28103, 28104, 28105]
COLORS = {
    "OCCLUSION": "#d93025",
    "CLEARANCE_OVERLAP": "#1a73e8",
    "BOTH": "#9334e6",
    "NONE": "#188038",
    "Reason": "#f9ab00",
    "Blocker": "#1a73e8",
    "Joint": "#9334e6",
}


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 3) if d else 0.0


def svg_start(w: int, h: int, title: str, subtitle: str = "") -> list[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#5f6368}.label{font-size:12px}.small{font-size:10px}.value{font-size:12px;font-weight:700}.grid{stroke:#e6e8eb;stroke-width:1}</style>',
        f'<text class="title" x="38" y="36">{escape(title)}</text>',
    ]
    if subtitle:
        out.append(f'<text class="sub" x="38" y="58">{escape(subtitle)}</text>')
    return out


def simple_bars(path: Path, title: str, labels: list[str], values: list[float], colors: list[str], subtitle: str = "") -> None:
    w, h, left, bottom, plot_h = 900, 480, 80, 380, 280
    out = svg_start(w, h, title, subtitle)
    for t in range(0, 101, 20):
        y = bottom - t / 100 * plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="860" y2="{y}"/>', f'<text class="label" x="44" y="{y+4}">{t}%</text>']
    gw = 730 / len(labels)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = 105 + i * gw + (gw-72)/2
        bh = value / 100 * plot_h
        out += [f'<rect x="{x}" y="{bottom-bh}" width="72" height="{bh}" rx="5" fill="{color}"/>', f'<text class="value" x="{x+36}" y="{bottom-bh-8}" text-anchor="middle">{value:.1f}%</text>', f'<text class="label" x="{x+36}" y="{bottom+24}" text-anchor="middle">{escape(label)}</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def confusion_svg(path: Path, confusion: dict[str, dict[str, int]]) -> None:
    w, h, left, top, cell = 830, 610, 255, 120, 105
    max_v = max(v for r in confusion.values() for v in r.values())
    out = svg_start(w, h, "C2 D4 reason confusion matrix", "Rows: GT · Columns: model output · count and row percentage")
    for j, pred in enumerate(REASONS):
        x = left+j*cell+cell/2
        out.append(f'<text class="small" x="{x}" y="{top-17}" text-anchor="middle" transform="rotate(-25 {x} {top-17})">{escape(pred)}</text>')
    for i, gt in enumerate(REASONS):
        y = top+i*cell
        total = sum(confusion[gt].values())
        out.append(f'<text class="small" x="{left-15}" y="{y+56}" text-anchor="end">{escape(gt)}</text>')
        for j, pred in enumerate(REASONS):
            v = confusion[gt][pred]
            intensity = v/max_v if max_v else 0
            color = f'rgb({int(246-95*intensity)},{int(248-75*intensity)},{int(252-20*intensity)})'
            x = left+j*cell
            stroke = '#188038' if gt == pred else '#fff'
            sw = 3 if gt == pred else 1
            out += [f'<rect x="{x}" y="{y}" width="{cell-5}" height="{cell-5}" rx="5" fill="{color}" stroke="{stroke}" stroke-width="{sw}"/>', f'<text class="value" x="{x+50}" y="{y+44}" text-anchor="middle">{v}</text>', f'<text class="small" x="{x+50}" y="{y+65}" text-anchor="middle">{pct(v,total):.1f}%</text>']
    out.append('</svg>')
    path.write_text("\n".join(out))


def grouped_bars(path: Path, title: str, labels: list[str], series: list[tuple[str,list[float],str]], subtitle: str = "") -> None:
    w, h, left, bottom, plot_h = 1080, 520, 75, 415, 300
    out = svg_start(w, h, title, subtitle)
    for t in range(0,101,20):
        y=bottom-t/100*plot_h
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="1040" y2="{y}"/>',f'<text class="label" x="40" y="{y+4}">{t}%</text>']
    gw=930/len(labels); bw=min(36,gw/(len(series)+1))
    for i,label in enumerate(labels):
        center=left+(i+.5)*gw
        for j,(_,vals,color) in enumerate(series):
            val=vals[i]; x=center+(j-(len(series)-1)/2)*(bw+5)-bw/2; bh=val/100*plot_h
            out.append(f'<rect x="{x}" y="{bottom-bh}" width="{bw}" height="{bh}" rx="3" fill="{color}"/>')
        out.append(f'<text class="small" x="{center}" y="{bottom+24}" text-anchor="middle">{escape(label.replace("_"," "))}</text>')
    lx=w-280
    for j,(name,_,color) in enumerate(series):
        x=lx+j*105; out += [f'<rect x="{x}" y="28" width="13" height="13" fill="{color}"/>',f'<text class="label" x="{x+18}" y="39">{escape(name)}</text>']
    out.append('</svg>'); path.write_text("\n".join(out))


def scene_bars(path: Path, scenes: list[dict]) -> None:
    w, h, left, right, top, rh = 1020, 1030, 245, 60, 70, 36
    out=svg_start(w,h,"C2 D4 reason accuracy by scene","Each scene: five seeds")
    for i,s in enumerate(sorted(scenes,key=lambda x:(-x['reason_pct'],x['scene_id']))):
        y=top+i*rh; v=s['reason_pct']; bw=(w-left-right)*v/100
        color='#188038' if v>=80 else '#1a73e8' if v>=50 else '#f9ab00' if v>0 else '#d93025'
        out += [f'<text class="small" x="{left-10}" y="{y+17}" text-anchor="end">{escape(s["scene_id"])}</text>',f'<rect x="{left}" y="{y}" width="{bw}" height="24" rx="4" fill="{color}"/>',f'<text class="value" x="{left+bw+8}" y="{y+17}">{s["reason_correct"]}/5 · {v:.0f}%</text>']
    out.append('</svg>'); path.write_text("\n".join(out))


def seed_grid(path: Path, scenes: list[dict]) -> None:
    w,h,left,top,cw,ch=960,990,285,80,112,34
    out=svg_start(w,h,"C2 D4 scene × seed reason outputs","Green border = correct reason; cell text = predicted label")
    abbrev={"OCCLUSION":"OCC","CLEARANCE_OVERLAP":"CLR","BOTH":"BOTH","NONE":"NONE"}
    for j,seed in enumerate(SEEDS):
        out.append(f'<text class="label" x="{left+j*cw+cw/2}" y="{top-12}" text-anchor="middle">{seed}</text>')
    for i,s in enumerate(sorted(scenes,key=lambda x:x['scene_id'])):
        y=top+i*ch
        out.append(f'<text class="small" x="{left-10}" y="{y+21}" text-anchor="end">{escape(s["scene_id"])}</text>')
        for j,o in enumerate(s['outputs']):
            pred=o['blocking_reason']; ok=pred==s['gt_reason']; x=left+j*cw
            color=COLORS[pred]; stroke='#0b8043' if ok else '#d93025'; sw=3 if ok else 2
            out += [f'<rect x="{x}" y="{y}" width="{cw-5}" height="{ch-4}" rx="3" fill="{color}" fill-opacity="0.82" stroke="{stroke}" stroke-width="{sw}"/>',f'<text class="small" x="{x+(cw-5)/2}" y="{y+20}" text-anchor="middle" fill="#fff">{abbrev[pred]}</text>']
    out.append('</svg>'); path.write_text("\n".join(out))


def decomposition_svg(path: Path, cats: dict) -> None:
    w,h=980,330
    items=[("Both correct",cats.get('both_correct',0),'#188038'),("Blocker only",cats.get('blocker_only',0),'#1a73e8'),("Reason only",cats.get('reason_only',0),'#f9ab00'),("Both wrong",cats.get('both_wrong',0),'#d93025')]
    out=svg_start(w,h,"C2 D4 reason–blocker decomposition","125 calls")
    x=45; y=105; total=sum(v for _,v,_ in items)
    for label,v,color in items:
        width=880*v/total
        if width:
            out += [f'<rect x="{x}" y="{y}" width="{width}" height="72" fill="{color}"/>',f'<text class="value" x="{x+width/2}" y="{y+31}" text-anchor="middle" fill="#fff">{v}</text>',f'<text class="small" x="{x+width/2}" y="{y+51}" text-anchor="middle" fill="#fff">{100*v/total:.1f}%</text>']
        x+=width
    lx=45
    for label,v,color in items:
        out += [f'<rect x="{lx}" y="{y+100}" width="12" height="12" fill="{color}"/>',f'<text class="label" x="{lx+18}" y="{y+111}">{label}</text>']; lx+=210
    out.append('</svg>'); path.write_text("\n".join(out))


def main() -> None:
    rows=[json.loads(x) for x in (EXP/'logs/runs.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['geometry_condition']=='C2' and r['direction_resolution']=='D4']
    gt={p.stem:json.loads(p.read_text())['L3'] for p in (EXP/'geometry/ground_truth').glob('*.json')}
    confusion={g:{p:0 for p in REASONS} for g in REASONS}
    pred_counts=Counter(); by_family=defaultdict(Counter); by_seed=defaultdict(Counter); by_scene=defaultdict(list)
    records=[]
    cats=Counter()
    for r in rows:
        truth=gt[r['scene_id']]; pred=r['parsed_response']; rc=bool(r['scoring']['blocking_reason_correct']); bc=bool(r['scoring']['blocker_valid_set_member'])
        confusion[truth['blocking_reason']][pred['blocking_reason']]+=1; pred_counts[pred['blocking_reason']]+=1
        for bucket in (by_family[r['scene_family']],by_seed[r['seed']]):
            bucket['n']+=1; bucket['reason']+=rc; bucket['blocker']+=bc; bucket['joint']+=rc and bc
        by_scene[r['scene_id']].append(r)
        cat='both_correct' if rc and bc else 'reason_only' if rc else 'blocker_only' if bc else 'both_wrong'; cats[cat]+=1
        records.append({'run_id':r['run_id'],'scene_id':r['scene_id'],'family':r['scene_family'],'seed':r['seed'],'target_object_id':r['target_object_id'],'gt_reason':truth['blocking_reason'],'valid_blocker_ids':truth['valid_blocker_ids'],'predicted_reason':pred['blocking_reason'],'predicted_blocker_id':pred['blocker_id'],'reason_correct':rc,'blocker_correct':bc,'joint_correct':rc and bc,'geometry':r['geometry_json_raw'],'raw_response':r['raw_response'],'image':str(EXP/r['numbered_rgb'])})
    scene_rows=[]
    for scene,rs in by_scene.items():
        rs=sorted(rs,key=lambda x:x['seed']); truth=gt[scene]
        scene_rows.append({'scene_id':scene,'family':rs[0]['scene_family'],'target_object_id':rs[0]['target_object_id'],'gt_reason':truth['blocking_reason'],'valid_blocker_ids':truth['valid_blocker_ids'],'reason_correct':sum(r['scoring']['blocking_reason_correct'] for r in rs),'reason_pct':pct(sum(r['scoring']['blocking_reason_correct'] for r in rs),5),'blocker_correct':sum(r['scoring']['blocker_valid_set_member'] for r in rs),'pred_counts':dict(Counter(r['parsed_response']['blocking_reason'] for r in rs)),'outputs':[{'seed':r['seed'],**r['parsed_response']} for r in rs]})
    class_metrics={}
    for reason in REASONS:
        tp=confusion[reason][reason]; actual=sum(confusion[reason].values()); predicted=sum(confusion[g][reason] for g in REASONS)
        precision=tp/predicted if predicted else 0; recall=tp/actual if actual else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
        class_metrics[reason]={'support':actual,'predicted':predicted,'correct':tp,'precision_pct':round(100*precision,3),'recall_pct':round(100*recall,3),'f1_pct':round(100*f1,3)}
    reason_correct=sum(x['reason_correct'] for x in records); blocker_correct=sum(x['blocker_correct'] for x in records); joint_correct=sum(x['joint_correct'] for x in records)
    result={'experiment_version':'l3_primary_blocker_v19','condition':'C2_D4','task':'blocking_reason and one strongest blocker','n':len(rows),'overall':{'reason_correct':reason_correct,'reason_pct':pct(reason_correct,len(rows)),'blocker_correct':blocker_correct,'blocker_pct':pct(blocker_correct,len(rows)),'joint_correct':joint_correct,'joint_pct':pct(joint_correct,len(rows))},'prediction_distribution':{r:{'n':pred_counts[r],'pct':pct(pred_counts[r],len(rows))} for r in REASONS},'confusion':confusion,'class_metrics':class_metrics,'joint_categories':dict(cats),'by_family':{f:{'n':by_family[f]['n'],'reason_correct':by_family[f]['reason'],'reason_pct':pct(by_family[f]['reason'],by_family[f]['n']),'blocker_correct':by_family[f]['blocker'],'blocker_pct':pct(by_family[f]['blocker'],by_family[f]['n']),'joint_correct':by_family[f]['joint'],'joint_pct':pct(by_family[f]['joint'],by_family[f]['n'])} for f in FAMILIES},'by_seed':{str(s):{'n':by_seed[s]['n'],'reason_correct':by_seed[s]['reason'],'reason_pct':pct(by_seed[s]['reason'],by_seed[s]['n']),'blocker_correct':by_seed[s]['blocker'],'blocker_pct':pct(by_seed[s]['blocker'],by_seed[s]['n']),'joint_correct':by_seed[s]['joint'],'joint_pct':pct(by_seed[s]['joint'],by_seed[s]['n'])} for s in SEEDS},'by_scene':sorted(scene_rows,key=lambda x:x['scene_id']),'records':records,'interpretation_limits':'The API-enforced output stores only final JSON, not hidden chain-of-thought. Decision patterns are inferred from supplied inputs and observed outputs.'}
    OUT.mkdir(parents=True,exist_ok=True); VIS.mkdir(parents=True,exist_ok=True)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'all_125_outputs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['run_id','scene_id','family','seed','target_object_id','gt_reason','valid_blocker_ids','predicted_reason','predicted_blocker_id','reason_correct','blocker_correct','joint_correct'])
        w.writeheader()
        for x in records:w.writerow({k:(json.dumps(x[k]) if k=='valid_blocker_ids' else x[k]) for k in w.fieldnames})
    (OUT/'report.md').write_text(f"# C2 D4 reason analysis\n\n- Calls: 125\n- Reason: {reason_correct}/125 = {pct(reason_correct,125):.1f}%\n- Blocker: {blocker_correct}/125 = {pct(blocker_correct,125):.1f}%\n- Joint: {joint_correct}/125 = {pct(joint_correct,125):.1f}%\n- OCCLUSION predictions: {pred_counts['OCCLUSION']}\n- Dominant error: pure OCCLUSION is never predicted; BOTH is often reduced to CLEARANCE_OVERLAP.\n")
    simple_bars(VIS/'00_overall_metrics.svg','C2 D4 overall accuracy',['Reason','Blocker','Joint'],[pct(reason_correct,125),pct(blocker_correct,125),pct(joint_correct,125)],[COLORS['Reason'],COLORS['Blocker'],COLORS['Joint']],'25 scenes × 5 seeds = 125 calls')
    simple_bars(VIS/'01_prediction_distribution.svg','C2 D4 reason output distribution',REASONS,[pct(pred_counts[r],125) for r in REASONS],[COLORS[r] for r in REASONS],'OCCLUSION was never emitted')
    confusion_svg(VIS/'02_confusion_matrix.svg',confusion)
    simple_bars(VIS/'03_gt_recall.svg','C2 D4 recall by ground-truth reason',REASONS,[class_metrics[r]['recall_pct'] for r in REASONS],[COLORS[r] for r in REASONS])
    grouped_bars(VIS/'04_family_metrics.svg','C2 D4 accuracy by scene family',[f.replace('_',' ') for f in FAMILIES],[('Reason',[result['by_family'][f]['reason_pct'] for f in FAMILIES],COLORS['Reason']),('Blocker',[result['by_family'][f]['blocker_pct'] for f in FAMILIES],COLORS['Blocker'])])
    grouped_bars(VIS/'05_seed_metrics.svg','C2 D4 accuracy by seed',[str(s) for s in SEEDS],[('Reason',[result['by_seed'][str(s)]['reason_pct'] for s in SEEDS],COLORS['Reason']),('Blocker',[result['by_seed'][str(s)]['blocker_pct'] for s in SEEDS],COLORS['Blocker'])])
    scene_bars(VIS/'06_scene_reason_accuracy.svg',scene_rows)
    seed_grid(VIS/'07_scene_seed_grid.svg',scene_rows)
    decomposition_svg(VIS/'08_reason_blocker_decomposition.svg',dict(cats))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
