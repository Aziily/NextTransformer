import type {Point,Run} from './types';

export const FINAL_STEP=46875;
export type RankingMetric='core'|'val';
export type RankingMode='final'|'replay';
export type LeaderboardEntry={run:Run;rank?:number;value?:number;core?:number;val?:number;latestCore?:Point;latestVal?:Point;differentBudget:boolean;coreOnly:boolean};

// Pick the most recently recorded attempt, never the attempt with the best score.
export function preferredRuns(runs:Run[],allAttempts=false):Run[]{
 if(allAttempts)return runs;
 const chosen=new Map<string,Run>();
 for(const run of runs){
  const previous=chosen.get(run.methodId);
  if(!previous||(run.lastRecordedAt??0)>(previous.lastRecordedAt??0))chosen.set(run.methodId,run);
 }
 return [...chosen.values()];
}

export function rankingCheckpoint(baseline:Run|undefined,metric:RankingMetric,cursor:number,runs:Run[]=[]):number|undefined{
 return (baseline?.[metric]??runs.flatMap(r=>r[metric])).filter(p=>Number.isFinite(p.value)&&p.step<=cursor&&p.step<=FINAL_STEP).map(p=>p.step).sort((a,b)=>a-b).at(-1);
}

// `coreOnly` marks runs that rank on CORE only (auto-research entries): they keep their measured values but take no
// place in the validation BPB ranking.
export function leaderboardEntries(runs:Run[],metric:RankingMetric,mode:RankingMode,step:number|undefined,coreOnly:(run:Run)=>boolean=()=>false):LeaderboardEntry[]{
 const at=mode==='final'?FINAL_STEP:step;
 const entries=runs.map(run=>{
  const point=(kind:RankingMetric)=>at===undefined?undefined:run[kind].find(p=>p.step===at&&Number.isFinite(p.value))?.value;
  const latest=(kind:RankingMetric)=>run[kind].filter(p=>Number.isFinite(p.value)&&(mode==='final'||(at!==undefined&&p.step<=at))).sort((a,b)=>a.step-b.step).at(-1);
  const target=Number(run.proof?.total_steps??FINAL_STEP),differentBudget=target!==FINAL_STEP;
  const core=point('core'),val=point('val');
  const only=coreOnly(run);
  return {run,core,val,value:(mode==='final'&&differentBudget)||(metric==='val'&&only)?undefined:metric==='core'?core:val,latestCore:latest('core'),latestVal:latest('val'),differentBudget,coreOnly:only};
 });
 entries.sort((a,b)=>{
  if(a.value===undefined&&b.value!==undefined)return 1;
  if(a.value!==undefined&&b.value===undefined)return -1;
  if(a.value!==undefined&&b.value!==undefined){const delta=metric==='core'?b.value-a.value:a.value-b.value;if(Math.abs(delta)>1e-12)return delta;}
  return a.run.methodId.localeCompare(b.run.methodId)||a.run.id.localeCompare(b.run.id);
 });
 let previous:number|undefined,rank=0;
 return entries.map((entry,index)=>{
  if(entry.value===undefined)return entry;
  if(previous===undefined||Math.abs(entry.value-previous)>1e-12)rank=index+1;
  previous=entry.value;
  return {...entry,rank};
 });
}
