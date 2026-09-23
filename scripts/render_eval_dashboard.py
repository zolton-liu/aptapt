#!/usr/bin/env python3
"""Render accumulated evaluation experiments as a standalone training dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_experiments(root: Path) -> list[dict]:
    experiments = []
    for path in sorted(root.glob("*/results.jsonl")):
        if path.parent.name.startswith("smoke"):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        experiments.append({"name": path.parent.name, "rows": rows})
    return experiments


def document(experiments: list[dict]) -> str:
    payload = json.dumps(experiments, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QFA 训练效果</title>
<style>
:root {{ color-scheme: light dark; --bg: light-dark(#fbfbfc,#17181a); --fg: light-dark(#17181a,#f1f2f3); --muted: light-dark(#68707a,#a8afb8); --line: light-dark(#d8dce1,#3a3e44); --good: #25a55f; --bad: #d54d4d; --accent: #4c78a8; --accent2: #f28e2b; }}
* {{ box-sizing:border-box }} body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg) }}
main {{ max-width:1180px; margin:auto; padding:28px 20px 48px }} h1,h2 {{ font-weight:600 }} h1 {{ margin:0 0 6px }} .sub {{ color:var(--muted); margin-bottom:26px }}
.metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:28px }} .metric {{ border-top:3px solid var(--accent); padding:12px 4px }} .metric strong {{ display:block; font-size:28px }} .metric span {{ color:var(--muted) }}
.notice {{ border-left:3px solid var(--accent2); padding:9px 12px; margin:0 0 24px; color:var(--muted); background:color-mix(in srgb,var(--accent2) 8%,transparent) }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:28px }} section {{ min-width:0 }} .wide {{ grid-column:1/-1 }} svg {{ display:block; width:100%; height:auto }} .frame {{ fill:none; stroke:var(--line) }} .axis {{ fill:var(--muted); font-size:12px }}
table {{ width:100%; border-collapse:collapse; font-size:14px }} th,td {{ padding:8px; border-bottom:1px solid var(--line); text-align:left }} th {{ color:var(--muted) }} .pass {{ color:var(--good) }} .fail {{ color:var(--bad) }}
@media(max-width:760px) {{ .metrics {{ grid-template-columns:1fr 1fr }} .grid {{ grid-template-columns:1fr }} .wide {{ grid-column:auto }} .table-wrap {{ overflow-x:auto }} }}
</style>
</head>
<body><main>
<h1>QFA 训练效果</h1><div class="sub">公开任务本地评测 · starter disabled · 单次采样</div>
<div class="notice">全量基线与定向回归的任务数量不同，不能把百分比直接当作同一测试集上的连续训练曲线。下方每个点均显示样本量 n。</div>
<div class="metrics" id="metrics"></div>
<div class="grid"><section class="wide"><h2>各次实验通过率（scope-aware）</h2><svg id="curve" viewBox="0 0 1100 320" aria-label="各次实验通过率"></svg></section>
<section><h2>最新版本分类表现</h2><svg id="categories" viewBox="0 0 540 330" aria-label="分类通过率"></svg></section>
<section><h2>失败类型</h2><svg id="failures" viewBox="0 0 540 330" aria-label="失败类型分布"></svg></section>
<section class="wide"><h2>逐题结果</h2><div class="table-wrap"><table><thead><tr><th>任务</th><th>分类</th><th>结果</th><th>模型调用</th><th>Tokens</th><th>耗时</th><th>失败类型</th></tr></thead><tbody id="tasks"></tbody></table></div></section></div>
</main><script>
const experiments={payload};
const latest=experiments.at(-1)||{{name:'无数据',rows:[]}}; const rows=latest.rows; const passed=rows.reduce((s,r)=>s+r.reward,0); const sum=k=>rows.reduce((s,r)=>s+(Number(r[k])||0),0);
const full=experiments.reduce((best,item)=>item.rows.length>(best?.rows.length||0)?item:best,null)||latest;
const fullPassed=full.rows.reduce((s,r)=>s+r.reward,0);
document.getElementById('metrics').innerHTML=[
  ['全量基线',`${{fullPassed}}/${{full.rows.length}} · ${{full.rows.length?(fullPassed/full.rows.length*100).toFixed(1):'—'}}%`],
  ['最新定向回归',rows.length?`${{passed}}/${{rows.length}} · ${{(passed/rows.length*100).toFixed(1)}}%`:'—'],
  ['最新回归范围',`${{rows.length}} 个任务`],
  ['最新总 Tokens',(sum('input_tokens')+sum('output_tokens')).toLocaleString()]
].map(x=>`<div class="metric"><strong>${{x[1]}}</strong><span>${{x[0]}}</span></div>`).join('');
function svgEl(tag,attrs={{}}){{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}} function text(svg,x,y,value,anchor='start'){{const e=svgEl('text',{{x,y,'text-anchor':anchor,class:'axis'}});e.textContent=value;svg.append(e)}}
function frame(svg,x,y,w,h){{svg.append(svgEl('rect',{{x,y,width:w,height:h,class:'frame'}}))}}
{{const svg=document.getElementById('curve'), shown=experiments.slice(-9),x0=70,y0=20,w=1000,h=220;frame(svg,x0,y0,w,h);const values=shown.map(e=>e.rows.reduce((s,r)=>s+r.reward,0)/e.rows.length);shown.forEach((e,i)=>{{const x=shown.length===1?x0+w/2:x0+i*w/(shown.length-1),y=y0+h-(values[i]*h);if(i){{const pv=values[i-1],px=shown.length===1?x:x0+(i-1)*w/(shown.length-1),py=y0+h-pv*h;svg.append(svgEl('line',{{x1:px,y1:py,x2:x,y2:y,stroke:'var(--line)','stroke-width':2,'stroke-dasharray':'5 5'}}))}}svg.append(svgEl('circle',{{cx:x,cy:y,r:6,fill:e===full?'var(--accent2)':'var(--accent)'}}));text(svg,x,y-12,(values[i]*100).toFixed(1)+'% · n='+e.rows.length,'middle');const short=e.name.length>20?e.name.slice(0,18)+'…':e.name;text(svg,x,y0+h+24,short,'middle')}});[0,.25,.5,.75,1].forEach(v=>text(svg,x0-10,y0+h-v*h,(v*100)+'%','end'))}}
function grouped(key){{const m=new Map;rows.forEach(r=>{{const k=r[key]||'unknown';if(!m.has(k))m.set(k,[]);m.get(k).push(r)}});return [...m]}}
function bars(id,data,value,color){{const svg=document.getElementById(id),x0=150,y0=20,w=350,h=270;frame(svg,x0,y0,w,h);const bh=Math.min(34,h/Math.max(1,data.length));data.forEach((d,i)=>{{const y=y0+i*bh+6,v=value(d),fill=typeof color==='function'?color(d):color;svg.append(svgEl('rect',{{x:x0,y,width:w*v,height:bh-10,fill}}));text(svg,x0-8,y+bh/2,d[0],'end');text(svg,x0+w*v+7,y+bh/2,(v*100).toFixed(0)+'%')}})}}
const cats=grouped('category');bars('categories',cats,d=>d[1].reduce((s,r)=>s+r.reward,0)/d[1].length,'var(--accent)');const fails=grouped('failure_type');bars('failures',fails,d=>d[1].length/Math.max(1,rows.length),d=>d[0]==='pass'?'var(--good)':'var(--bad)');
document.getElementById('tasks').innerHTML=rows.map(r=>`<tr><td>${{r.task_id}}</td><td>${{r.category}}</td><td class="${{r.reward?'pass':'fail'}}">${{r.reward?'PASS':'FAIL'}}</td><td>${{r.model_calls}}</td><td>${{((r.input_tokens||0)+(r.output_tokens||0)).toLocaleString()}}</td><td>${{Number(r.duration_sec||0).toFixed(1)}}s</td><td>${{r.failure_type}}</td></tr>`).join('');
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("eval_runs"))
    parser.add_argument("--output", type=Path, default=Path("reports/training-effect.html"))
    args = parser.parse_args()
    experiments = load_experiments(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document(experiments), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
