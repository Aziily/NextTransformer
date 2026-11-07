import {useState} from 'react';
import {ArrowLeft,FlaskConical} from 'lucide-react';
import {asset} from './lib/links';
import './arena-example.css';

const models = ['Opus','GLM','DeepSeek'];
const candidates = [
 {name:'dense_full',bpb:0.80135,delta:0.00104,status:'Control'},
 {name:'attnres',bpb:0.80031,delta:0,status:'Control'},
 {name:'attnres_full_canon',bpb:0.79849,delta:-0.00182,status:'Research winner'},
 {name:'attnres_canon_abcd',bpb:0.79819,delta:-0.00212,status:'Research winner'},
 {name:'attnres_full_canon_abcd',bpb:0.79651,delta:-0.00380,status:'Research winner'},
];

export default function ArenaExample(){
 const [winnersOnly,setWinnersOnly]=useState(false);
 const visible=candidates.filter(c=>!winnersOnly||c.status==='Research winner');
 return <section className="arena arena-example" aria-label="Arena results example">
  <div className="arena-hero">
   <div><p className="eyebrow">03 / ARENA · RESULTS EXAMPLE</p><h2>Independent research. Equal compute.<span>One harness. Three models. Their own experiments.</span></h2>
    <p className="arena-lede">Opus, GLM and DeepSeek each choose what to build and which experiments to run. Every session shares the same initial protocol and total research compute allowance.</p>
    <p className="arena-development"><FlaskConical size={14}/>Example only · no live sessions</p>
    <div className="arena-actions"><a className="secondary" href={asset('#arena')}><ArrowLeft size={15}/> Back to Arena</a></div>
   </div>
   <div className="arena-principles">
    <div><span>HARNESS</span><b>Claude Code</b><p>Gym MCP · model and client versions pinned before launch.</p></div>
    <div><span>PER MODEL</span><b>1.96 × 10²⁰ FLOPs</b><p>One shared allowance for training, evaluation, checks and failed attempts.</p></div>
    <div><span>MODEL CALLS</span><b>No spending cap</b><p>Token use and actual cost are recorded.</p></div>
    <div><span>SCHEDULING ESTIMATE</span><b>~89 hours per model</b><p>~710 GPU-hours on a comparable eight-GPU setup. Formal training is separate.</p></div>
   </div>
  </div>
  <div className="arena-surface">
   <div className="arena-surface-heading"><div><h3>Research sessions</h3><span>Three planned sessions · no scores yet</span></div><span className="arena-reference">Total research cap · 5.88 × 10²⁰ FLOPs</span></div>
   <div className="arena-example-table-wrap"><table className="arena-example-table"><thead><tr><th scope="col">Model</th><th scope="col">Harness</th><th scope="col">Final candidate</th><th scope="col">Used FLOPs</th><th scope="col">Final CORE ↑</th><th scope="col">Status</th></tr></thead><tbody>{models.map(model=><tr key={model}><th scope="row">{model}</th><td>Claude Code</td><td>—</td><td>—</td><td>—</td><td><span className="arena-status admitted">Planned</span></td></tr>)}</tbody></table></div>
   <div className="arena-caption"><span>Missing scores are not zeroes. Only completed, audited formal runs enter the final ranking.</span><span>Architecture and search strategy are chosen by each model.</span></div>
  </div>
  <div className="arena-lower">
   <div className="arena-panel"><h3>Research results · historical example</h3><p>Selected Arena v1 candidates at update 2,000. These illustrate the display, not a required experiment schedule for new sessions.</p>
    <label className="arena-example-filter"><input type="checkbox" checked={winnersOnly} onChange={e=>setWinnersOnly(e.target.checked)}/> Winners only</label>
    <div className="arena-example-table-wrap"><table className="arena-example-table"><thead><tr><th scope="col">Candidate</th><th scope="col">Dev BPB ↓</th><th scope="col">Δ vs attnres</th></tr></thead><tbody>{visible.map(c=><tr key={c.name}><th scope="row"><code>{c.name}</code></th><td>{c.bpb.toFixed(5)}</td><td>{c.delta===0?'—':`${c.delta<0?'−':'+'}${Math.abs(c.delta).toFixed(5)}`}</td></tr>)}</tbody></table></div>
    <p className="arena-example-caption" aria-live="polite">{visible.length} selected candidates · not the full v1 population</p>
   </div>
   <div className="arena-panel"><h3>Budget & evidence</h3><p>Historical measurements inform the allocation; model sessions have not started.</p>
    <details open><summary>Where does the budget come from?</summary><p>V1 search used 549.7 GPU-hours, and transfer checks 42.1. Adding 20% headroom to 591.8 gives approximately 710 GPU-hours per model.</p><p>The FLOPs allowance is estimated as 20 anchor-equivalent 2,000-update runs, based on v1's 17 attempts with headroom. This is a total budget, not a prescribed experiment count or length.</p></details>
    <details><summary>What can each model decide?</summary><p>The initial architecture, controls, experiment lengths, candidate count, promotions and search strategy. Training, checks and failed attempts draw from one research allocation.</p></details>
    <details><summary>Historical source and final results</summary><p>The supplied Arena v1 progress report, November 4 snapshot, pages 2–3. Its formal run was incomplete and unaudited. No interim formal score is presented as final here.</p><p>A connected result will include its configuration, code revision, trajectory, disclosure and audit.</p></details>
   </div>
  </div>
 </section>;
}
