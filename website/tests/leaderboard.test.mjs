import assert from 'node:assert/strict';
import {test} from 'node:test';
import {FINAL_STEP,leaderboardEntries,preferredRuns,rankingCheckpoint} from '../app/lib/leaderboard.ts';

const run=(id,core,val,extra={})=>({id,methodId:id,cohort:'external',name:id,step:FINAL_STEP,core,val,...extra});
const a=run('a',[{step:8000,value:.9},{step:FINAL_STEP,value:.3}],[{step:8000,value:.8},{step:FINAL_STEP,value:.7}],{finalBpb:.01});
const b=run('b',[{step:8000,value:.2},{step:FINAL_STEP,value:.4}],[{step:8000,value:.9},{step:FINAL_STEP,value:.75}]);
const partial=run('partial',[{step:8000,value:.99}],[{step:8000,value:.1}],{step:8000,status:'partial'});

test('final leaderboards use final periodic checkpoints, correct directions, and no best-so-far or separate final pass',()=>{
 const core=leaderboardEntries([a,b,partial],'core','final',undefined);
 assert.deepEqual(core.map(e=>[e.run.id,e.rank]),[['b',1],['a',2],['partial',undefined]]);
 const val=leaderboardEntries([a,b,partial],'val','final',undefined);
 assert.deepEqual(val.map(e=>[e.run.id,e.rank]),[['a',1],['b',2],['partial',undefined]]);
 assert.equal(val[0].value,.7);
 assert.equal(val[2].latestVal.value,.1);
 assert.equal(partial.val.length,1);
});

test('replay ranks only exact shared steps and never future observations',()=>{
 assert.equal(rankingCheckpoint(a,'core',7999),undefined);
 assert.equal(rankingCheckpoint(a,'core',12000),8000);
 const early=leaderboardEntries([a,b,partial],'core','replay',8000);
 assert.equal(early[0].run.id,'partial');
 const gap=leaderboardEntries([a,b,partial],'core','replay',16000);
 assert.ok(gap.every(e=>e.rank===undefined));
 assert.ok(gap.every(e=>e.latestCore.step===8000));
 const before=leaderboardEntries([a,b,partial],'core','replay',undefined);
 assert.ok(before.every(e=>e.rank===undefined&&e.latestCore===undefined));
});

test('ties share ranks; invalid final scores and different budgets stay unranked',()=>{
 const tied=run('tie',[{step:FINAL_STEP,value:.4}],[]);
 const invalid=run('invalid',[{step:FINAL_STEP,value:NaN}],[]);
 const otherBudget=run('other-budget',[{step:FINAL_STEP,value:.8}],[],{proof:{total_steps:100000}});
 const entries=leaderboardEntries([a,b,tied,invalid,otherBudget],'core','final');
 assert.deepEqual(entries.filter(e=>e.rank).map(e=>e.rank),[1,1,3]);
 assert.equal(entries.find(e=>e.run.id==='invalid').rank,undefined);
 assert.equal(entries.find(e=>e.run.id==='other-budget').rank,undefined);
});

test('experiment preference unifies sources by latest record without selecting the best score',()=>{
 const older={...a,methodId:'same',cohort:'native-v3',lastRecordedAt:10};
 const newer={...b,methodId:'same',cohort:'external',lastRecordedAt:20};
 assert.deepEqual(preferredRuns([older,newer,partial]).map(r=>r.id),['b','partial']);
 assert.equal(preferredRuns([older,newer],true).length,2);
});

test('a core-only run ranks on CORE and is left out of the validation BPB ranking',()=>{
 const only=run=>run.id==='a';
 const core=leaderboardEntries([a,b],'core','final',undefined,only);
 assert.deepEqual(core.map(e=>[e.run.id,e.rank]),[['b',1],['a',2]]);
 const val=leaderboardEntries([a,b],'val','final',undefined,only);
 assert.deepEqual(val.map(e=>[e.run.id,e.rank]),[['b',1],['a',undefined]]);
 assert.equal(val[1].val,.7);
 assert.ok(val[1].coreOnly&&!val[0].coreOnly);
});
