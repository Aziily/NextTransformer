import assert from 'node:assert/strict';
import {test} from 'node:test';
import {LEVELS,arenaRows,coreOnly,originOf,provenance,researcherName,researchers} from '../app/lib/arena.ts';
import {fullDenseBaseline} from '../app/lib/baseline.ts';
import {FINAL_STEP,leaderboardEntries,preferredRuns} from '../app/lib/leaderboard.ts';
import {arena,methods,runs} from './results.mjs';

const ranks=(pool,metric)=>new Map(leaderboardEntries(pool,metric,'final',undefined,run=>coreOnly(run.methodId,arena())).flatMap(e=>e.rank===undefined?[]:[[e.run.id,e.rank]]));
const published=()=>{const all=runs(),pool=preferredRuns(all);return {all,pool,baseline:fullDenseBaseline(all),ranks:{core:ranks(pool,'core'),val:ranks(pool,'val')}};};

test('every published card names a model card and its runs, with a known involvement level',()=>{
 const cards=arena(),known=new Set(methods().map(m=>m.id)),ids=new Set(runs().map(r=>r.id));
 assert.ok(cards.length>=1);
 for(const card of cards){
  assert.ok(known.has(card.methodId),card.id);
  assert.ok(card.runIds.every(id=>ids.has(id)),card.id);
  assert.ok(card.researcher.involvement in LEVELS,card.id);
  assert.equal(card.arenaProtocol,'arena-v1');
 }
});

test('Circle Loop is the legacy L3 example and keeps its measured leaderboard place',()=>{
 const {all,baseline,ranks:r}=published(),[row]=arenaRows(arena().filter(c=>c.id==='circle-loop'),all,methods(),baseline,r);
 assert.equal(row.entry.status,'legacy');
 assert.equal(row.entry.researcher.involvement,'L3');
 assert.equal(row.run.step,FINAL_STEP);
 assert.equal(row.core,row.run.core.find(p=>p.step===FINAL_STEP).value);
 assert.ok(Math.abs(row.coreDelta-(row.core-baseline.core.find(p=>p.step===FINAL_STEP).value))<1e-12);
 assert.equal(row.rankCore,r.core.get(row.run.id));
 assert.equal(researcherName(row.entry),'ChatGPT + Codex');
});

test('provenance marks arena methods only, and rows without a run stay unranked',()=>{
 const cards=arena();
 assert.equal(originOf('circle',cards),'agent');
 assert.equal(originOf('dense_full',cards),'human');
 assert.equal(provenance('gdn',cards),undefined);
 const pending={...cards[0],id:'pending',methodId:'pending',runIds:[],status:'admitted',submitted:'2026-12-01'};
 const {all,baseline,ranks:r}=published(),rows=arenaRows([...cards,pending],all,methods(),baseline,r);
 const row=rows.find(x=>x.entry.id==='pending');
 assert.equal(row.run,undefined);
 assert.equal(row.rankCore,undefined);
 assert.equal(row.coreDelta,undefined);
 const order=['complete','admitted','legacy','retracted'],statuses=rows.map(x=>x.entry.status);
 assert.deepEqual(statuses,[...statuses].sort((a,b)=>order.indexOf(a)-order.indexOf(b)));
 const groups=researchers(rows),legacy=groups.find(g=>g.rows.some(x=>x.entry.id==='circle-loop'));
 assert.equal(groups.reduce((n,g)=>n+g.rows.length,0),rows.length);
 assert.ok(legacy.levels.includes('L3'));
});

test('arena entries rank on CORE only and keep their measured validation BPB',()=>{
 const cards=arena(),{all,pool,baseline,ranks:r}=published();
 for(const card of cards.filter(c=>c.status==='complete')){
  assert.ok(coreOnly(card.methodId,cards),card.id);
  const [row]=arenaRows([card],all,methods(),baseline,r);
  assert.equal(row.rankCore,r.core.get(row.run.id));
  assert.equal(row.rankVal,undefined);
  assert.equal(row.val,row.run.val.find(p=>p.step===FINAL_STEP).value);
  const val=leaderboardEntries(pool,'val','final',undefined,run=>coreOnly(run.methodId,cards)).find(e=>e.run.id===row.run.id);
  assert.ok(val.coreOnly&&val.rank===undefined&&val.val===row.val);
 }
 assert.equal(coreOnly('circle',cards),false);
 assert.notEqual(r.val.get('circle'),undefined);
});
