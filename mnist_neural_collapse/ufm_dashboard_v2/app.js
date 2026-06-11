const DATA = window.UFM_DATA;
const $ = (id)=>document.getElementById(id);
const fmt=(x,d=3)=> (x===null||x===undefined||Number.isNaN(x))?'—':Number(x).toFixed(d).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1');
const methods = DATA.summary.method_order;
const display = DATA.summary.method_display;
const desc = DATA.summary.method_desc;
const dims = DATA.summary.prefix_dims.map(String);
const colors = ['#94a3b8','#ffe37b','#59c2ff','#ff8f9c','#b69cff','#ffa45c','#22c55e','#82f0a1','#60a5fa','#e879f9','#14b8a6'];
const colorMap = {}; methods.forEach((m,i)=>colorMap[m]=colors[i%colors.length]);
const metricInfo = {
  accuracy:['Accuracy','higher'], ce:['Cross-entropy','lower'], effective_rank_H:['Effective rank H','higher'], numerical_rank_H:['Numerical rank H','higher'],
  nc2_etf_error_H:['ETF error H','lower'], nc2_etf_error_W:['ETF error W','lower'], nc3_align_mean:['NC3 alignment','higher'], self_duality_error:['Self-duality error','lower'],
  spherical_margin_H:['Spherical margin H','higher'], spherical_margin_W:['Spherical margin W','higher'], logit_margin_mean:['Mean logit margin','higher'],
  H_norm_mean:['Mean H norm','context'], W_norm_mean:['Mean W norm','context'], mrl_ce_total:['MRL CE total','lower'], l2_loss:['L2 loss','lower'], vicreg_var_loss:['VICReg var loss','lower']
};
const primaryMetrics=['effective_rank_H','nc2_etf_error_H','nc3_align_mean','spherical_margin_H','ce','accuracy','self_duality_error','logit_margin_mean','H_norm_mean'];
const animMetrics=['effective_rank_H','nc2_etf_error_H','nc3_align_mean','spherical_margin_H','ce','accuracy','logit_margin_mean'];
let anim={idx:0, timer:null, playing:false};

function row(method, dim, mode){
  dim=Number(dim); if(method==='single-scale') mode='single'; else mode=mode||'mrl';
  return DATA.final.find(r=>r.method===method && r.mode===mode && Number(r.dim)===dim);
}
function histRows(method, dim){
  dim=Number(dim); const mode=method==='single-scale'?'single':'mrl';
  return DATA.history.filter(r=>r.method===method && r.mode===mode && Number(r.dim)===dim).sort((a,b)=>a.epoch-b.epoch);
}
function selectedMethods(){return Array.from(document.querySelectorAll('.methodCheck:checked')).map(x=>x.value)}
function metricLabel(m){return (metricInfo[m]||[m])[0]}
function metricBetter(m){return (metricInfo[m]||['','context'])[1]}

function setup(){
  renderHero(); renderStory(); setupControls(); renderAll();
}
function renderHero(){
  const s=DATA.summary, c=s.comparisons||{};
  $('heroStats').innerHTML = [
    [`${s.num_methods}`, 'methods/runs'], [`${s.num_history_rows}`, 'history rows'], [`+${fmt(c.rank_gain_var_vs_uniform,2)}`, 'rank gain: Block-Var vs vanilla'], [`${fmt(c.etf_reduction_var_vs_uniform_pct,1)}%`, 'ETF error reduction']
  ].map(x=>`<div class="heroStat"><b>${x[0]}</b><span>${x[1]}</span></div>`).join('');
}
function renderStory(){
  const u=DATA.summary.uniform_d32, v=DATA.summary.varonly_d32, ol=DATA.summary.onlylarge_d32;
  const c=DATA.summary.comparisons||{};
  const cards=[
    ['Diagnosis','Vanilla MRL rank-saturates',`d32 effective rank ends at ${fmt(u.effective_rank_H,2)} although the ideal NC rank is ${DATA.summary.ideal_rank}.`,fmt(u.effective_rank_H,2)],
    ['Control','Full-only recovers ETF',`Only-large reaches rank ${fmt(ol.effective_rank_H,2)} with essentially zero ETF error.`,fmt(ol.nc2_etf_error_H,5)],
    ['Fix','Block-Var reactivates geometry',`Block-Var raises d32 rank by ${fmt(c.rank_gain_var_vs_uniform,2)} and strongly lowers ETF error.`,fmt(v.effective_rank_H,2)],
    ['Mechanism','Variance, not covariance',`Cov-only does not revive zero-variance blocks. The variance floor directly penalizes dead blocks.`,`${fmt(c.etf_reduction_var_vs_uniform_pct,1)}%`]
  ];
  $('storyCards').innerHTML=cards.map(c=>`<div class="storyCard"><span class="tag">${c[0]}</span><h3>${c[1]}</h3><div class="bigNum">${c[3]}</div><p>${c[2]}</p></div>`).join('');
  $('methodMap').innerHTML=methods.map(m=>`<div class="methodBox"><div class="methodName"><span class="swatch" style="background:${colorMap[m]}"></span>${display[m]||m}</div><p>${desc[m]||''}</p></div>`).join('');
}
function setupControls(){
  for(const m of primaryMetrics){ $('metricSelect').add(new Option(metricLabel(m),m)); }
  $('metricSelect').value='effective_rank_H';
  for(const d of dims){ ['dimSelect','animDim','gramDim'].forEach(id=>$(id).add(new Option('d = '+d,d))); }
  $('dimSelect').value='32'; $('animDim').value='32'; $('gramDim').value='32';
  for(const m of methods){
    const lab=document.createElement('label');
    lab.innerHTML=`<input class="methodCheck" type="checkbox" value="${m}"><span><b style="color:${colorMap[m]}">${display[m]}</b><br><small class="muted">${desc[m]||''}</small></span>`;
    $('methodChecks').appendChild(lab);
  }
  setMethods(['single-scale','only-large','uniform','large-heavy','var-only','var-cov','cov-only','only-small+big']);
  $('coreBtn').onclick=()=>{setMethods(['single-scale','only-large','uniform','large-heavy','var-only','cov-only','only-small+big']); renderAll();};
  $('allBtn').onclick=()=>{setMethods(methods); renderAll();};
  $('clearBtn').onclick=()=>{setMethods([]); renderAll();};
  document.querySelectorAll('select,.methodCheck').forEach(el=>el.addEventListener('change',()=>{ if(el.id && el.id.startsWith('anim')) resetAnimation(false); renderAll(); }));
  $('imgRun').add(new Option('All runs','all'));
  for(const m of methods){ if(DATA.history.some(r=>r.method===m)) $('animRun').add(new Option(display[m],m)); $('gramRun').add(new Option(display[m],m)); $('imgRun').add(new Option(display[m],m)); }
  $('animRun').value='var-only'; $('gramRun').value='var-only'; $('imgRun').value='var-only';
  for(const m of animMetrics){ $('animMetric').add(new Option(metricLabel(m),m)); } $('animMetric').value='nc2_etf_error_H';
  ['all','final','history','gram','gap','other'].forEach(c=>$('imgCat').add(new Option(c,c))); $('imgCat').value='all';
  $('playBtn').onclick=toggleAnimation; $('resetBtn').onclick=()=>resetAnimation(true); $('epochSlider').oninput=e=>{anim.idx=Number(e.target.value); renderAnimation();};
  $('imgSearch').oninput=renderImages; $('tableSearch').oninput=renderTables;
}
function setMethods(list){const set=new Set(list); document.querySelectorAll('.methodCheck').forEach(x=>x.checked=set.has(x.value));}
function renderAll(){renderLine(); renderBar(); renderEvidence(); renderAnimation(); renderBlocks(); renderGrams(); renderImages(); renderTables();}

function getCtx(canvas){ const r=canvas.getBoundingClientRect(); const dpr=window.devicePixelRatio||1; canvas.width=r.width*dpr; canvas.height=r.height*dpr; const ctx=canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0); return [ctx,r.width,r.height]; }
function scales(points,w,h,pad,xKey='x',yKey='y',forceZero=false){
  let xs=points.map(p=>+p[xKey]).filter(Number.isFinite), ys=points.map(p=>+p[yKey]).filter(Number.isFinite);
  let xmin=Math.min(...xs), xmax=Math.max(...xs), ymin=Math.min(...ys), ymax=Math.max(...ys);
  if(forceZero) ymin=Math.min(0,ymin); if(xmin===xmax){xmin-=1;xmax+=1;} if(ymin===ymax){ymin-=1;ymax+=1;}
  const yspan=ymax-ymin, xspan=xmax-xmin; ymin-=yspan*.08; ymax+=yspan*.08; xmin-=xspan*.02; xmax+=xspan*.02;
  return {xmin,xmax,ymin,ymax,x:x=>pad+(x-xmin)/(xmax-xmin)*(w-2*pad), y:y=>h-pad-(y-ymin)/(ymax-ymin)*(h-2*pad)};
}
function axes(ctx,w,h,pad,sc,xlab='',ylab=''){
  ctx.clearRect(0,0,w,h); ctx.strokeStyle='rgba(255,255,255,.18)'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(pad,h-pad); ctx.lineTo(w-pad,h-pad); ctx.lineTo(w-pad,pad); ctx.stroke();
  ctx.fillStyle='#9aabc9'; ctx.font='12px system-ui'; ctx.textAlign='center'; [sc.xmin, sc.xmax].forEach(x=>ctx.fillText(fmt(x,0),sc.x(x),h-pad+20)); ctx.textAlign='right'; [sc.ymin, sc.ymax].forEach(y=>ctx.fillText(fmt(y,2),pad-8,sc.y(y)+4));
  ctx.fillStyle='#dbeafe'; ctx.textAlign='left'; if(ylab)ctx.fillText(ylab,pad,pad-15); if(xlab){ctx.textAlign='right'; ctx.fillText(xlab,w-pad,h-10);}
}
function drawLineChart(canvas, series, metric, highlightEpoch=null){
  const [ctx,w,h]=getCtx(canvas), pad=54; const pts=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p.y)); if(!pts.length){ctx.clearRect(0,0,w,h);return;}
  const sc=scales(pts,w,h,pad,'x','y',metric==='accuracy'); axes(ctx,w,h,pad,sc,'dimension / epoch',metricLabel(metric));
  for(const s of series){ctx.strokeStyle=s.color;ctx.lineWidth=3;ctx.beginPath();s.points.forEach((p,i)=>{const x=sc.x(p.x),y=sc.y(p.y);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle=s.color;s.points.forEach(p=>{ctx.beginPath();ctx.arc(sc.x(p.x),sc.y(p.y),4,0,Math.PI*2);ctx.fill();});}
  if(highlightEpoch!==null){ const hp=pts.reduce((a,p)=>Math.abs(p.x-highlightEpoch)<Math.abs(a.x-highlightEpoch)?p:a,pts[0]); ctx.strokeStyle='rgba(255,255,255,.35)';ctx.setLineDash([6,6]);ctx.beginPath();ctx.moveTo(sc.x(hp.x),pad);ctx.lineTo(sc.x(hp.x),h-pad);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='white';ctx.beginPath();ctx.arc(sc.x(hp.x),sc.y(hp.y),6,0,Math.PI*2);ctx.fill(); }
}
function drawBarChart(canvas, items, metric){
  const [ctx,w,h]=getCtx(canvas), pad=62; const vals=items.map(i=>i.value).filter(Number.isFinite); if(!vals.length){ctx.clearRect(0,0,w,h);return;} const ymax=Math.max(...vals)*1.18 || 1; const ymin=Math.min(0,Math.min(...vals));
  ctx.clearRect(0,0,w,h); ctx.strokeStyle='rgba(255,255,255,.18)'; ctx.beginPath(); ctx.moveTo(pad,h-pad); ctx.lineTo(w-pad,h-pad); ctx.stroke(); ctx.fillStyle='#9aabc9'; ctx.font='12px system-ui'; ctx.textAlign='right'; ctx.fillText(fmt(ymax,2),pad-8,pad+4); ctx.fillText('0',pad-8,h-pad+4);
  const bw=(w-2*pad)/items.length*.68; items.forEach((it,i)=>{const x=pad+(i+.5)*(w-2*pad)/items.length-bw/2; const bh=(it.value-ymin)/(ymax-ymin)*(h-2*pad); const y=h-pad-bh; ctx.fillStyle=it.color; ctx.fillRect(x,y,bw,bh); ctx.fillStyle='#eef6ff'; ctx.textAlign='center'; ctx.font='11px system-ui'; ctx.save(); ctx.translate(x+bw/2,h-pad+16); ctx.rotate(-.35); ctx.fillText(it.label,0,0); ctx.restore(); ctx.fillText(fmt(it.value,3),x+bw/2,y-7);});
  ctx.fillStyle='#dbeafe'; ctx.textAlign='left'; ctx.font='12px system-ui'; ctx.fillText(metricLabel(metric),pad,pad-17);
}
function renderLegend(id, series){ $(id).innerHTML=series.map(s=>`<span><i style="background:${s.color}"></i>${s.label}</span>`).join(''); }

function renderLine(){
  const metric=$('metricSelect').value; const ms=selectedMethods();
  $('metricHelp').textContent='better: '+metricBetter(metric);
  const series=ms.map(m=>({label:display[m], color:colorMap[m], points:DATA.final.filter(r=>r.method===m && (m==='single-scale'?r.mode==='single':r.mode==='mrl') && r[metric]!=null).sort((a,b)=>a.dim-b.dim).map(r=>({x:r.dim,y:r[metric]}))}));
  drawLineChart($('lineChart'),series,metric); renderLegend('lineLegend',series);
}
function renderBar(){
  const metric=$('metricSelect').value, dim=$('dimSelect').value; $('barHelp').textContent=`d=${dim}, better: ${metricBetter(metric)}`;
  const items=selectedMethods().map(m=>{const r=row(m,dim); return r&&r[metric]!=null?{label:(display[m]||m).replace(' MRL','').replace('Independent ','Single '),value:r[metric],color:colorMap[m]}:null}).filter(Boolean);
  drawBarChart($('barChart'),items,metric);
}
function badge(val,metric){ const b=metricBetter(metric); if(val==null)return ''; let good=false,warn=false; if(metric==='effective_rank_H') good=val>8; else if(metric==='nc2_etf_error_H') good=val<0.05, warn=val<0.18; else if(metric==='nc3_align_mean') good=val>0.97,warn=val>0.9; else if(metric==='spherical_margin_H') good=val>0.85,warn=val>0.55; const cls=good?'statusGood':warn?'statusWarn':'statusBad'; return `<span class="statusBadge ${cls}">${fmt(val,3)}</span>`; }
function renderEvidence(){
  const rows=methods.map(m=>row(m,32)).filter(Boolean); const cols=['method','accuracy','effective_rank_H','nc2_etf_error_H','nc3_align_mean','spherical_margin_H','self_duality_error'];
  let html='<table><thead><tr><th>Method</th><th>Acc</th><th>Rank</th><th>ETF err</th><th>NC3</th><th>Margin</th><th>Self-duality</th><th>Interpretation</th></tr></thead><tbody>';
  for(const r of rows){
    let interp=''; if(r.method==='uniform')interp='rank-saturated vanilla MRL'; else if(r.method==='var-only')interp='best MRL fix: reactivates blocks'; else if(r.method==='only-large')interp='full-only control, recovers ETF'; else if(r.method==='cov-only')interp='covariance alone does not help'; else if(r.method==='single-scale')interp='independent target geometry'; else interp=desc[r.method]||'';
    html+=`<tr><td><span class="swatch" style="background:${colorMap[r.method]}"></span>${display[r.method]}</td><td>${fmt(r.accuracy,3)}</td><td>${badge(r.effective_rank_H,'effective_rank_H')}</td><td>${badge(r.nc2_etf_error_H,'nc2_etf_error_H')}</td><td>${badge(r.nc3_align_mean,'nc3_align_mean')}</td><td>${badge(r.spherical_margin_H,'spherical_margin_H')}</td><td>${fmt(r.self_duality_error,3)}</td><td class="muted" style="text-align:left">${interp}</td></tr>`;
  }
  html+='</tbody></table>'; $('evidenceTable').innerHTML=html;
}

function resetAnimation(toStart){stopAnimation(); if(toStart)anim.idx=0; syncAnim(); renderAnimation();}
function stopAnimation(){ if(anim.timer){clearInterval(anim.timer);anim.timer=null;} anim.playing=false; $('playBtn').textContent='Play'; }
function toggleAnimation(){ if(anim.playing){stopAnimation();return;} const rows=histRows($('animRun').value,$('animDim').value); if(rows.length<2)return; anim.playing=true; $('playBtn').textContent='Pause'; const speed=Number($('animSpeed').value); anim.timer=setInterval(()=>{const r=histRows($('animRun').value,$('animDim').value); if(anim.idx>=r.length-1){stopAnimation();return;} anim.idx++; syncAnim(); renderAnimation();},speed); }
function syncAnim(){const rows=histRows($('animRun').value,$('animDim').value); if(anim.idx>rows.length-1)anim.idx=Math.max(0,rows.length-1); $('epochSlider').max=Math.max(0,rows.length-1); $('epochSlider').value=anim.idx; $('epochFinal').textContent=rows.length?rows[rows.length-1].epoch:'—';}
function renderAnimation(){
  const method=$('animRun').value, dim=$('animDim').value, metric=$('animMetric').value; const rows=histRows(method,dim); if(!rows.length)return; syncAnim(); const cur=rows[anim.idx], fin=rows[rows.length-1]; $('epochNow').textContent=cur.epoch; $('frameNow').textContent=`${anim.idx+1} / ${rows.length}`;
  const cards=[['accuracy',cur.accuracy,fin.accuracy],['ce',cur.ce,fin.ce],['effective_rank_H',cur.effective_rank_H,fin.effective_rank_H],['nc2_etf_error_H',cur.nc2_etf_error_H,fin.nc2_etf_error_H],['nc3_align_mean',cur.nc3_align_mean,fin.nc3_align_mean],['spherical_margin_H',cur.spherical_margin_H,fin.spherical_margin_H]];
  $('animCards').innerHTML=cards.map(c=>`<div class="metricCard"><div class="label">${metricLabel(c[0])}</div><div class="value">${fmt(c[1],3)}</div><div class="sub">final ${fmt(c[2],3)}</div></div>`).join('');
  const pts=rows.filter(r=>r[metric]!=null).map(r=>({x:r.epoch,y:r[metric]})); drawLineChart($('animMetricChart'),[{label:display[method],color:colorMap[method],points:pts}],metric,cur.epoch); $('animCaption').innerHTML=`${display[method]}, d=${dim}, epoch ${cur.epoch}. ${metricLabel(metric)} = <b>${fmt(cur[metric],4)}</b>.`;
  drawPhase($('phaseChart'),rows,cur,'effective_rank_H','nc2_etf_error_H','rank','ETF error'); drawPhase($('rankNc3Chart'),rows,cur,'effective_rank_H','nc3_align_mean','rank','NC3');
}
function drawPhase(canvas, rows, cur, xk, yk, xlab, ylab){ const pts=rows.filter(r=>r[xk]!=null&&r[yk]!=null).map(r=>({x:r[xk],y:r[yk],epoch:r.epoch})); const [ctx,w,h]=getCtx(canvas), pad=54; if(!pts.length){ctx.clearRect(0,0,w,h);return;} const sc=scales(pts,w,h,pad); axes(ctx,w,h,pad,sc,xlab,ylab); ctx.strokeStyle='rgba(255,255,255,.22)';ctx.lineWidth=2;ctx.beginPath();pts.forEach((p,i)=>{if(i)ctx.lineTo(sc.x(p.x),sc.y(p.y));else ctx.moveTo(sc.x(p.x),sc.y(p.y));});ctx.stroke(); const idx=Math.max(0,pts.findIndex(p=>p.epoch===cur.epoch)); const prog=pts.slice(0,idx+1); ctx.strokeStyle=colorMap[$('animRun').value];ctx.lineWidth=3;ctx.beginPath();prog.forEach((p,i)=>{if(i)ctx.lineTo(sc.x(p.x),sc.y(p.y));else ctx.moveTo(sc.x(p.x),sc.y(p.y));});ctx.stroke(); pts.forEach((p,i)=>{ctx.fillStyle=i===idx?'white':'rgba(255,255,255,.45)';ctx.beginPath();ctx.arc(sc.x(p.x),sc.y(p.y),i===idx?6:2.5,0,Math.PI*2);ctx.fill();}); }

function renderBlocks(){
  const ms=selectedMethods().filter(m=>m!=='single-scale'); const blocks=[...new Set(DATA.blocks.map(b=>b.block))];
  drawGroupedBars($('blockStdChart'), blocks, ms, 'std_mean', 'Mean block std'); drawGroupedBars($('blockNormChart'), blocks, ms, 'norm_mean', 'Mean block norm'); renderLegend('blockLegend', ms.map(m=>({label:display[m],color:colorMap[m]}))); drawCoordStd();
}
function drawGroupedBars(canvas, groups, ms, key, title){ const [ctx,w,h]=getCtx(canvas), pad=58; ctx.clearRect(0,0,w,h); const vals=DATA.blocks.filter(b=>ms.includes(b.method)).map(b=>b[key]).filter(Number.isFinite); const ymax=Math.max(1,...vals)*1.15; ctx.strokeStyle='rgba(255,255,255,.18)';ctx.beginPath();ctx.moveTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke(); const gw=(w-2*pad)/groups.length; const bw=gw/Math.max(ms.length,1)*.72; groups.forEach((g,gi)=>{ms.forEach((m,mi)=>{const r=DATA.blocks.find(b=>b.method===m&&b.block===g); if(!r)return; const x=pad+gi*gw+mi*(gw/ms.length)+(gw/ms.length-bw)/2; const bh=r[key]/ymax*(h-2*pad); ctx.fillStyle=colorMap[m]; ctx.fillRect(x,h-pad-bh,bw,bh);}); ctx.fillStyle='#9aabc9';ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillText(g,pad+gi*gw+gw/2,h-pad+18);}); ctx.fillStyle='#dbeafe';ctx.textAlign='left';ctx.fillText(title,pad,pad-17); }
function drawCoordStd(){ const ms=selectedMethods().filter(m=>m!=='single-scale'); const series=ms.map(m=>({label:display[m],color:colorMap[m],points:DATA.coordStats.filter(r=>r.method===m).map(r=>({x:r.coord,y:r.std}))})); drawLineChart($('coordStdChart'),series,'coordinate std'); }

function renderGrams(){ const m=$('gramRun').value, d=$('gramDim').value; drawHeat('gramA',DATA.grams[m]?.mrl?.[d]); drawHeat('gramB',DATA.grams[m]?.single?.[d]); $('gramTitleA').textContent=`${display[m]} Gram`; $('gramCapA').textContent=`MRL class-mean cosine Gram, d=${d}`; $('gramCapB').textContent=`Independent single-scale class-mean cosine Gram, d=${d}`; }
function heatColor(v,min,max){ const t=(v-min)/(max-min||1); const r=Math.round(35+220*t), g=Math.round(80+100*(1-Math.abs(t-.5)*2)), b=Math.round(240*(1-t)+35); return `rgb(${r},${g},${b})`; }
function drawHeat(id,mat){ const el=$(id); if(!mat){el.innerHTML='<div class="muted">No matrix</div>';return;} const vals=mat.flat(); const min=Math.min(...vals), max=Math.max(...vals); el.innerHTML=mat.flatMap(row=>row.map(v=>`<div class="heatCell" style="background:${heatColor(v,min,max)}" title="${fmt(v,5)}">${fmt(v,2)}</div>`)).join(''); }

function renderImages(){ const run=$('imgRun').value, cat=$('imgCat').value, q=$('imgSearch').value.toLowerCase(); let imgs=DATA.images.filter(i=>(run==='all'||i.method===run)&&(cat==='all'||i.category===cat)&&(`${i.title} ${i.filename} ${i.method}`.toLowerCase().includes(q))); $('imageGallery').innerHTML=imgs.map(i=>`<div class="imgCard"><img src="${i.path}" loading="lazy"><div class="imgMeta"><b>${i.title}</b><small>${display[i.method]} · ${i.category} · ${i.filename}</small></div></div>`).join('') || '<p class="muted">No images match the filters.</p>'; }

function renderTables(){ const q=$('tableSearch').value.toLowerCase(); renderFinalTable(q); renderGapTable(q); renderConfigTable(q); }
function renderFinalTable(q){ const cols=['method','mode','dim','accuracy','ce','effective_rank_H','nc2_etf_error_H','nc3_align_mean','self_duality_error','spherical_margin_H','H_norm_mean','W_norm_mean','mrl_loss_weight','vicreg_var_loss','vicreg_cov_loss']; const rows=DATA.final.filter(r=>JSON.stringify(r).toLowerCase().includes(q)); $('finalTable').innerHTML=makeTable(rows,cols); }
function renderGapTable(q){ const cols=['method','dim','gram_fro_distance','gram_fro_distance_normalized']; const rows=DATA.gaps.filter(r=>JSON.stringify(r).toLowerCase().includes(q)); $('gapTable').innerHTML=makeTable(rows,cols); }
function renderConfigTable(q){ const rows=Object.entries(DATA.configs).map(([method,cfg])=>({method,display:display[method],loss_weights:cfg.loss_weight_values||cfg.loss_weights,vicreg:cfg.vicreg||'none',alpha:cfg.alpha,beta:cfg.beta,epochs:cfg.epochs,lr:cfg.lr,details:JSON.stringify(cfg)})).filter(r=>JSON.stringify(r).toLowerCase().includes(q)); $('configTable').innerHTML=makeTable(rows,['method','display','loss_weights','vicreg','alpha','beta','epochs','lr']); }
function makeTable(rows,cols){ return '<table><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>{let v=r[c]; if(c==='method')v=display[v]||v; if(typeof v==='number')v=fmt(v,5); return `<td>${v??'—'}</td>`}).join('')+'</tr>').join('')+'</tbody></table>'; }

window.addEventListener('resize',()=>renderAll());
setup();
