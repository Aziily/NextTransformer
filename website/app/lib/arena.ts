import type {ArenaEntry,Method,Run} from './types';

export const FINAL_STEP=46875;
export const LEVELS:Record<ArenaEntry['researcher']['involvement'],{name:string;note:string}>={
 L0:{name:'Fully autonomous',note:'People start the task and provide compute, then do not intervene until submission.'},
 L1:{name:'Operations only',note:'People fix environments, clusters or launches and make no research decision.'},
 L2:{name:'Human-guided',note:"People choose among the agent's proposals or give directional hints."},
 L3:{name:'Human-led',note:'People lead the idea or write method code; the agents assist.'}};
export const STATUS:Record<ArenaEntry['status'],string>={admitted:'In training',complete:'Ranked entry',legacy:'Legacy example',retracted:'Retracted'};
export const VERIFICATION:Record<ArenaEntry['verification'],string>={'self-reported':'Self-reported',reproduced:'Reproduced',maintainer:'Maintainer run'};
export type Origin='all'|'human'|'agent';
export type Ranks={core:Map<string,number>;val:Map<string,number>};
export type ArenaRow={entry:ArenaEntry;run?:Run;method?:Method;core?:number;val?:number;coreDelta?:number;valDelta?:number;rankCore?:number;rankVal?:number};

// The arena entry behind a method: the provenance every leaderboard row shows.
export const provenance=(methodId:string,entries:ArenaEntry[])=>entries.find(e=>e.methodId===methodId);
export const originOf=(methodId:string,entries:ArenaEntry[]):Exclude<Origin,'all'>=>provenance(methodId,entries)?'agent':'human';
// Auto-research entries under the arena rules rank on CORE only, a test the research never ran on its own candidates;
// their validation BPB is shown but not ranked. Legacy and retracted cards keep the ordinary leaderboard treatment.
export const coreOnly=(methodId:string,entries:ArenaEntry[])=>entries.some(e=>e.methodId===methodId&&(e.status==='admitted'||e.status==='complete'));
export const researcherName=(entry:ArenaEntry)=>entry.researcher.llms.map(m=>m.name).join(' + ')+(entry.researcher.framework.name==='none'?'':' · '+entry.researcher.framework.name);
const final=(run:Run|undefined,kind:'core'|'val')=>run?.[kind].find(p=>p.step===FINAL_STEP&&Number.isFinite(p.value))?.value;
const statusOrder:ArenaEntry['status'][]=['complete','admitted','legacy','retracted'];

// One row per entry: its latest named run, final measured scores, differences from the accepted
// Dense baseline at the same step, and the run's main-leaderboard ranks. Ranks are supplied by the
// caller so this view never ranks differently from the leaderboard.
export function arenaRows(entries:ArenaEntry[],runs:Run[],methods:Method[],baseline:Run|undefined,ranks:Ranks):ArenaRow[]{
 return entries.map(entry=>{
  const run=runs.filter(r=>entry.runIds.includes(r.id)).sort((a,b)=>(b.lastRecordedAt??0)-(a.lastRecordedAt??0))[0];
  const core=final(run,'core'),val=final(run,'val'),denseCore=final(baseline,'core'),denseVal=final(baseline,'val');
  return {entry,run,method:methods.find(m=>m.id===entry.methodId),core,val,
   coreDelta:core!==undefined&&denseCore!==undefined?core-denseCore:undefined,valDelta:val!==undefined&&denseVal!==undefined?val-denseVal:undefined,
   rankCore:run&&ranks.core.get(run.id),rankVal:run&&ranks.val.get(run.id)};
 }).sort((a,b)=>statusOrder.indexOf(a.entry.status)-statusOrder.indexOf(b.entry.status)||(a.rankCore??Infinity)-(b.rankCore??Infinity)||b.entry.submitted.localeCompare(a.entry.submitted));
}

// Entries grouped by researcher (LLMs and framework). Budgets are disclosed, not controlled, so this is descriptive.
export function researchers(rows:ArenaRow[]){
 const groups=new Map<string,{name:string;framework:string;rows:ArenaRow[]}>();
 for(const row of rows){
  const name=researcherName(row.entry),group=groups.get(name)??{name,framework:row.entry.researcher.framework.name,rows:[]};
  group.rows.push(row);groups.set(name,group);
 }
 return [...groups.values()].map(g=>({...g,levels:[...new Set(g.rows.map(r=>r.entry.researcher.involvement))].sort(),
  bestCore:Math.max(...g.rows.map(r=>r.core??-Infinity)),ranked:g.rows.filter(r=>r.entry.status==='complete').length}));
}
