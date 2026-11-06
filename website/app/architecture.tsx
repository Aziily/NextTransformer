'use client';

import { useId, type CSSProperties, type ReactNode } from 'react';
import CircleArchitecture from './circle-architecture';
import { fitLabels, nodeWidth } from './lib/node-labels';

type ArchitectureProps = { methodId: string; large?: boolean; nodes?: string[]; loop?: boolean };
const C = {
  ink: '#494E52', muted: '#68717A', line: '#B8C5D2',
  blue: '#527FA8', blueLight: '#EAF2FA', purple: '#8A6CB2', purpleLight: '#F1ECF8',
  rose: '#C2798A', roseLight: '#FBEEF2', gray: '#F3F6F9', white: '#FFFFFF',
};

const DESCRIPTIONS: Record<string, [string, string]> = {
  "sssl": ["Local, local, local, full", "The historical SSSL reference alternates short and full causal layers."],
  "dense_full": ["Full causal attention at every depth", "The Dense Baseline: 23 full-attention layers calibrated to the shared budget."],
  "dense": [
    "Independent depth",
    "Each block owns a different set of parameters."
  ],
  "ut_act": [
    "Adaptive Computation Time",
    "Halting mass accumulates separately for every token."
  ],
  "loop": [
    "Fixed recursion, shared weights",
    "Two passes through exactly the same nine core blocks."
  ],
  "huginn": [
    "A recurrent latent workspace",
    "The input is re-injected; the recurrent state keeps evolving."
  ],
  "ouro": [
    "Read out at multiple depths",
    "All candidate readouts contribute to the training objective."
  ],
  "deeploop": [
    "Stabilize repeated depth",
    "A shared full stack with a different residual parameterization."
  ],
  "mor": [
    "Route at recursion boundaries",
    "A token either executes a whole next round or leaves the core."
  ],
  "loopformer": [
    "Condition a fixed two-pass path",
    "The same core receives different time / step-size modulation."
  ],
  "circle": [
    "A ring of reusable blocks",
    "After the complete first round, every block can be an exit."
  ],
  "dsa": [
    "Select tokens before attention",
    "A lightweight indexer scores individual visible keys."
  ],
  "qsa": [
    "Pool four tokens into one key",
    "Score compressed keys, then expand selected blocks back to tokens."
  ],
  "nsa": [
    "Three branches, gated together",
    "Each branch retains a different part of the context."
  ],
  "seer": [
    "Learn a map of important blocks",
    "A learned block gate predicts where attention is useful."
  ],
  "sparse": [
    "Query-dependent block routing",
    "Different queries route to different historical KV blocks."
  ],
  "deltanet": [
    "Correct the stored value",
    "Update memory using a prediction error, not just another KV sum."
  ],
  "state": [
    "Forget, write, and read",
    "An input-dependent gate decides what the matrix should retain."
  ],
  "kda": [
    "Fine-grained gates + delta update",
    "First decay individual features, then correct the retained memory."
  ],
  "retnet": [
    "A spectrum of memory horizons",
    "Different heads retain information over different time scales."
  ],
  "elu_qknorm": ["A fixed q/k scale", "RMS-normalized, temperature-scaled q/k feed positive features; the state is read without a denominator and normalized per head."],
  "gdn": ["Scalar gate + delta update", "One learned decay per head shrinks the memory; the delta rule then corrects it."],
  "gsa_qknorm": ["Gated slot memory", "Keys and values are written into 64 gated slots; each query reads them back through a softmax over slots."],
  "mqa": ["One shared key/value head", "All twelve query heads read the same key and value projections."],
  "gqa": ["Three key/value groups", "Each key/value head serves four query heads."],
  "mla_qknorm": ["Attention through a compressed latent", "Keys and values are rebuilt from a 256-dimensional latent; q and k are RMS-normalized."],
  "mla_nope_qknorm": ["Latent attention without rotation", "No rotary key: order enters only through the causal mask."],
  "residual_standard": ["Plain Pre-Norm residual", "Every sublayer adds F(RMSNorm(h)) to one unscaled stream."],
  "attnres": ["Attention over depth", "Each sublayer reads a softmax mixture of the embedding and earlier group sums."],
  "gdn_hybrid": ["Three gated delta layers, then full attention", "Eighteen GDN layers carry a compact state; every fourth layer reads the full causal prefix."],
  "gdn_hybrid_swa": ["Gated delta and windowed attention, alternating", "Eleven GDN layers alternate with eleven attention layers; at 2K context the 2048-key window covers every causal key."],
  "custom": ["Custom architecture", "A contributor-provided mechanism. Follow the arrows to trace the computation."]
};

/** Mechanism illustrations, rather than operator graphs. Counts are labeled
 * explicitly; token grids and memory matrices are intentionally schematic. */
export default function Architecture({ methodId, large = false, nodes, loop = true }: ArchitectureProps) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '');
  const arrowId = `${uid}-arrow`;
  const label = methodId.replaceAll('_', ' ');
  const figure = methodId !== 'custom' && methodId in DESCRIPTIONS ? methodId : nodes?.length ? 'custom' : methodId;
  const description = DESCRIPTIONS[figure === 'custom' ? 'custom' : ({fixed_recursive:'loop',gla:'state',moba:'sparse'} as Record<string,string>)[figure] ?? figure] ?? DESCRIPTIONS.custom;
  if (methodId === 'circle' && !nodes?.length) return <CircleArchitecture large={large}/>;

  function Text({ x, y, children, size = 14, color = C.ink, anchor = 'start', weight = 500, ...rest }: {
    x: number; y: number; children: ReactNode; size?: number; color?: string;
    anchor?: 'start' | 'middle' | 'end'; weight?: number; opacity?: number;
  }) {
    return <text x={x} y={y} fill={color} fontSize={size} fontWeight={weight} textAnchor={anchor} {...rest}>{children}</text>;
  }
  function Box({ x, y, w = 120, h = 52, title, sub, tone = 'blue', muted = false }: {
    x: number; y: number; w?: number; h?: number; title: string; sub?: string;
    tone?: 'blue' | 'purple' | 'rose'; muted?: boolean;
  }) {
    const color = C[tone], fill = C[`${tone}Light`];
    return <g opacity={muted ? .37 : 1}>
      <rect x={x} y={y} width={w} height={h} rx={7} fill={fill} stroke={color} strokeWidth={1.2}/>
      <Text x={x+w/2} y={y+h/2+(sub ? -3 : 5)} size={14} anchor="middle" weight={600}>{title}</Text>
      {sub && <Text x={x+w/2} y={y+h/2+16} size={12} anchor="middle" color={C.muted}>{sub}</Text>}
    </g>;
  }
  function Path({ d, color = C.blue, dashed = false, flow = false, arrow = true, width = 2, opacity = 1 }: {
    d: string; color?: string; dashed?: boolean; flow?: boolean; arrow?: boolean; width?: number; opacity?: number;
  }) {
    return <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round"
      opacity={opacity} strokeDasharray={dashed ? '5 5' : flow ? '8 6' : undefined}
      className={flow ? 'arch-flow' : undefined} markerEnd={arrow ? `url(#${arrowId})` : undefined}/>;
  }
  function Particle({ d, color = C.purple, delay = 0, duration = 4 }: { d: string; color?: string; delay?: number; duration?: number }) {
    return <circle r={4} fill={color} className="arch-moving">
      <animateMotion path={d} dur={`${duration}s`} begin={`${delay}s`} repeatCount="indefinite"/>
    </circle>;
  }
  function Caption({ children, x = 300, y = 254 }: { children: ReactNode; x?: number; y?: number }) {
    return <Text x={x} y={y} size={13} anchor="middle" color={C.muted}>{children}</Text>;
  }
  function Matrix({ x, y, cell = 17, rows = 4, cols = 4, tone = 'blue', faded = false, values }: {
    x: number; y: number; cell?: number; rows?: number; cols?: number; tone?: 'blue' | 'purple' | 'rose'; faded?: boolean; values?: number[];
  }) {
    return <g>{Array.from({ length: rows * cols }, (_, i) => <rect key={i}
      x={x+(i%cols)*(cell+3)} y={y+Math.floor(i/cols)*(cell+3)} width={cell} height={cell} rx={3}
      fill={C[tone]} opacity={(values?.[i] ?? (.22+((i*7+3)%10)*.065))*(faded ? .4 : 1)}/>)}</g>;
  }
  function TokenRow({ x, y, n = 8, selected, tone = 'blue', size = 24 }: { x: number; y: number; n?: number; selected?: number[]; tone?: 'blue'|'purple'|'rose'; size?: number }) {
    return <g>{Array.from({length:n}, (_,i) => {
      const active = !selected || selected.includes(i);
      return <g key={i}><rect x={x+i*(size+6)} y={y} width={size} height={size} rx={5}
        fill={active ? C[`${tone}Light`] : C.gray} stroke={active ? C[tone] : '#DDE4EB'} strokeWidth={active ? 1.7 : 1}/>
        <Text x={x+i*(size+6)+size/2} y={y+size/2+4} anchor="middle" size={11} color={active ? C[tone] : '#A6B2BF'}>{i+1}</Text></g>;
    })}</g>;
  }
  function SharedBlocks({ x, y, count = 3, tone = 'purple' }: {x:number;y:number;count?:number;tone?:'blue'|'purple'|'rose'}) {
    return <g>{Array.from({length:count},(_,i)=><g key={i}>
      <Box x={x+i*78} y={y} w={60} h={42} title={count===3?['B₁','…','B₉'][i]:`B${i+1}`} tone={tone}/>
      {i<count-1&&<Path d={`M${x+i*78+61} ${y+21}H${x+(i+1)*78-7}`} color={C[tone]}/>}
    </g>)}</g>;
  }

  let body: ReactNode;
  let viewBox = methodId==='mla_qknorm'||methodId==='mla_nope_qknorm'?'0 25 600 330':'0 35 600 250';
  switch (figure) {
    case 'sssl':
    case 'dense_full':
      body = <>
        {[0,1,2,3].map(i=><g key={i}>
          <rect x={38+i*137} y={100} width={100} height={80} rx={12} fill={methodId==='sssl'&&i<3?C.blueLight:C.purpleLight} stroke={methodId==='sssl'&&i<3?C.blue:C.purple} strokeWidth={1.4}/>
          <Text x={88+i*137} y={131} size={17} anchor="middle" weight={700}>{methodId==='sssl'&&i<3?'Short':'Full'}</Text>
          <Text x={88+i*137} y={155} size={12} anchor="middle" color={C.muted}>causal attention</Text>
          {i<3&&<Path d={`M${141+i*137} 140H${171+i*137}`} color={C.blue}/>}
        </g>)}
        <Caption y={221}>{methodId==='sssl'?'SSSL repeats across 24 independent blocks':'Every layer sees the complete causal prefix'}</Caption>
        <Caption y={252}>{methodId==='sssl'?'Historical run IDs and checkpoints are preserved':'23 layers · FFN 6144 · calibrated to the shared budget'}</Caption>
      </>;
      break;
    case 'dense':
      body = <>
        {[0,1,2,3,4,5].map((i) => <g key={i}>
          <rect x={37+i*89} y={89} width={63} height={100} rx={12} fill={i%2 ? C.blueLight : '#F0F5FB'} stroke={C.blue} strokeWidth={1.4}/>
          <Text x={68.5+i*89} y={132} size={17} anchor="middle" weight={700}>{i===3?'…':`L${i<3?i+1:i+19}`}</Text>
          <Text x={68.5+i*89} y={155} size={12} anchor="middle" color={C.muted}>{i===3?'middle':`θ${i<3?i+1:i+19}`}</Text>
          {i<5&&<Path d={`M${102+i*89} 139H${120+i*89}`} color={C.blue}/ >}
        </g>)}
        <Path d="M38 214H545" color={C.line} arrow={false}/>
        <Text x={40} y={235} size={13} color={C.muted}>Input representation</Text><Text x={546} y={235} size={13} color={C.muted} anchor="end">LM readout</Text>
        <Caption y={264}>24 physical blocks · no weight sharing</Caption>
        <Particle d="M38 139H545" color={C.blue}/>
      </>;
      break;
    case 'ut_act':
      body = <>
        {[1,2,3].map((r,i)=><Text key={r} x={164+i*135} y={77} size={12} anchor="middle" color={C.muted}>RECURRENCE {r}</Text>)}
        {[0,1,2].map((lane)=><g key={lane}>
          <Text x={28} y={110+lane*48} size={13} color={C.muted}>{`token ${'ABC'[lane]}`}</Text>
          {[0,1,2].map(i=><g key={i}>
            <rect x={122+i*135} y={88+lane*48} width={84} height={32} rx={8}
              fill={i<=lane?C[['blue','purple','rose'][lane]+'Light' as keyof typeof C]:C.gray}
              stroke={i===lane?C[['blue','purple','rose'][lane] as keyof typeof C]:i<lane?C.line:'#E2E7EC'}
              strokeDasharray={i>lane?'3 4':undefined}/>
            <Text x={164+i*135} y={109+lane*48} anchor="middle" size={12} color={i>lane?'#9CA8B6':C.ink}>{i===lane?'halt':i<lane?'update':'frozen'}</Text>
          </g>)}
          <Path d={`M207 ${104+lane*48}H250 M342 ${104+lane*48}H385`} color={C[['blue','purple','rose'][lane] as keyof typeof C]} dashed={lane===0} arrow={false}/>
        </g>)}
        <Text x={534} y={77} size={12} anchor="middle" color={C.muted}>OUTPUT</Text>
        {[0,1,2].map(lane=><g key={lane}>
          <Path d={`M480 ${104+lane*48}H500`} color={C.purple}/>
          <Text x={534} y={109+lane*48} size={13} anchor="middle" color={C.purple} weight={600}>Σ pₜhₜ</Text>
        </g>)}
        <Caption y={247}>Weighted state mixture · remainder + ponder cost</Caption>
        <Caption y={270}>The reference backend still evaluates all candidate passes.</Caption>
      </>;
      break;
    case 'fixed_recursive':
    case 'loop':
      body = <>
        <Box x={24} y={120} w={89} h={54} title="Prelude" sub="3 blocks" tone="blue"/>
        <Path d="M115 147H133V102H155" color={C.blue}/>
        <Text x={157} y={74} size={12} color={C.muted}>PASS 1</Text><SharedBlocks x={155} y={81}/>
        <Text x={157} y={177} size={12} color={C.muted}>PASS 2</Text><SharedBlocks x={155} y={184}/>
        <Path d="M373 102C440 102 440 158 137 158V205H148" color={C.purple} flow/>
        <Path d="M373 205H449V147H474" color={C.purple}/>
        <Box x={481} y={120} w={95} h={54} title="Coda" sub="3 blocks" tone="blue"/>
        <Text x={405} y={88} size={12} color={C.purple}>same θ</Text>
        <Caption y={264}>3 + 9 × 2 + 3 · full BPTT · no routing or conditioner</Caption>
        <Particle d="M373 102C440 102 440 158 137 158V205H373" duration={6}/>
      </>;
      break;
    case 'huginn':
      body = <>
        <Box x={26} y={77} w={128} h={54} title="Input features" sub="fixed prelude" tone="blue"/>
        <Box x={224} y={108} w={119} h={60} title="Adapter" sub="concat + norm" tone="purple"/>
        <Box x={416} y={108} w={153} h={60} title="Shared core" sub="8 blocks / pass" tone="purple"/>
        <Path d="M155 104H186V129H216" color={C.blue}/>
        <Path d="M345 138H408" color={C.purple} flow/>
        <Path d="M492 170V211H283V176" color={C.purple} flow/>
        <Particle d="M492 170V211H283V176" duration={4}/>
        <Box x={26} y={173} w={135} h={48} title="Initial state" sub="row-keyed noise" tone="rose"/>
        <Path d="M163 195H197V151H216" color={C.rose}/>
        <Text x={397} y={198} size={12} anchor="middle" color={C.purple}>evolving state sₜ</Text>
        {[0,1,2,3].map(i=><rect key={i} x={406+i*35} y={67} width={28} height={16} rx={4} fill={i<2?C.gray:C.purpleLight} stroke={i<2?C.line:C.purple} strokeDasharray={i<2?'3 3':undefined}/>)}
        <Text x={476} y={99} anchor="middle" size={11} color={C.muted}>last ≤2 passes backpropagate</Text>
        <Caption>Sample 1–4 passes per update · early passes still cost forward FLOPs</Caption>
      </>;
      break;
    case 'ouro':
      body = <>
        {[0,1,2].map(i=><g key={i}>
          <Box x={36+i*190} y={81} w={144} h={53} title={`Core · pass ${i+1}`} sub="shared parameters" tone="purple"/>
          {i<2&&<Path d={`M${182+i*190} 107H${218+i*190}`} color={C.purple} flow/>}
          <Path d={`M${108+i*190} 136V159`} color={C.rose}/>
          <Box x={43+i*190} y={166} w={130} h={42} title="Coda + head" tone="rose"/>
          <Text x={108+i*190} y={230} size={14} anchor="middle" color={C.rose}>{i===2?'remaining mass':'p'+(i+1)+' · CE'+(i+1)}</Text>
          <circle cx={157+i*190} cy={80} r={8} fill={C.white} stroke={C.rose}/>
        </g>)}
        <Caption y={264}>Weighted CE − entropy → final 1,024 updates train gates only</Caption>
      </>;
      break;
    case 'deeploop':
      body = <>
        <rect x={107} y={77} width={434} height={119} rx={16} fill="#FBF9FE" stroke="#D6C7E9" strokeDasharray="5 5"/>
        <Text x={132} y={98} size={12} color={C.purple}>ONE ATTENTION / MLP SUBLAYER</Text>
        <Text x={27} y={151} size={18} weight={700}>x</Text>
        <Path d="M44 145H127" color={C.purple}/>
        <Box x={134} y={118} w={101} h={52} title="RMSNorm" tone="purple"/>
        <Path d="M237 144H260" color={C.purple}/>
        <Box x={267} y={118} w={88} h={52} title="Fβ" sub="scaled init" tone="purple"/>
        <Path d="M357 144H386" color={C.purple}/>
        <circle cx={407} cy={144} r={18} fill={C.roseLight} stroke={C.rose}/><Text x={407} y={150} size={21} anchor="middle" color={C.rose}>+</Text>
        <Path d="M63 144V62H407V118" color={C.rose}/>
        <Text x={287} y={58} size={13} color={C.rose} anchor="middle">α · x</Text>
        <Path d="M427 144H452" color={C.purple}/>
        <Box x={459} y={118} w={96} h={52} title="RMSNorm" tone="purple"/>
        <Path d="M489 199C489 231 178 231 149 200" color={C.purple} flow/>
        <Text x={320} y={218} size={13} anchor="middle" color={C.purple}>12-block stack × 2</Text>
        <Caption y={263}>α = √(2N) · β = 1/√(8N) · no separate prelude / coda</Caption>
      </>;
      break;
    case 'mor':
      body = <>
        {[0,1,2].map(round=><g key={round}>
          <rect x={44+round*190} y={84} width={132} height={133} rx={14} fill={C.purpleLight} stroke={C.purple}/>
          <Text x={110+round*190} y={107} size={14} anchor="middle" weight={700}>Round {round+1}</Text>
          {Array.from({length:9},(_,i)=><circle key={i} cx={76+round*190+(i%3)*34} cy={133+Math.floor(i/3)*24}
            r={7} fill={i<[9,6,3][round]?C.purple:C.white} stroke={i<[9,6,3][round]?C.purple:'#D2C6E0'} opacity={i<[9,6,3][round]?1:.5}/ >)}
          <Text x={110+round*190} y={202} anchor="middle" size={12} color={C.purple}>9 shared blocks</Text>
          {round<2&&<><Path d={`M${177+round*190} 148H${226+round*190}`} color={C.rose} flow/>
            <Text x={202+round*190} y={129} size={11} anchor="middle" color={C.rose}>router</Text></>}
          <Text x={110+round*190} y={241} size={14} anchor="middle" color={C.purple}>{['100%','64%','32%'][round]} target survival</Text>
        </g>)}
        <Caption y={269}>Illustrative token sets · exited tokens never re-enter the core</Caption>
      </>;
      break;
    case 'loopformer':
      body = <>
        <Box x={48} y={74} w={206} h={43} title="t = 0 · Δt = ½" tone="rose"/>
        <Box x={347} y={74} w={206} h={43} title="t = ½ · Δt = ½" tone="rose"/>
        <Box x={48} y={158} w={206} h={56} title="Shared core · pass 1" sub="9 blocks" tone="purple"/>
        <Box x={347} y={158} w={206} h={56} title="Shared core · pass 2" sub="same 9 blocks" tone="purple"/>
        <Path d="M151 119V150 M450 119V150" color={C.rose} flow/>
        <Text x={162} y={141} size={12} color={C.rose}>modulate</Text><Text x={461} y={141} size={12} color={C.rose}>modulate</Text>
        <Path d="M256 186H338" color={C.purple} flow/>
        <Text x={298} y={169} size={12} color={C.purple} anchor="middle">state</Text>
        <Caption y={247}>Bounded multiplier: 1 + 0.5 · tanh(offset)</Caption>
        <Caption y={267}>Fixed2 adaptation · no adaptive stopping or consistency loss</Caption>
      </>;
      break;
    case 'dsa':
      body=<>
        <Box x={27} y={76} w={109} h={49} title="Query q" tone="blue"/>
        <Path d="M138 101H199" color={C.blue}/>
        <Box x={206} y={70} w={161} h={60} title="ReLU indexer" sub="multi-head scores" tone="purple"/>
        <Path d="M369 101H426" color={C.purple}/>
        <Box x={433} y={76} w={133} h={49} title="Causal top-k" tone="rose"/>
        <Text x={28} y={169} size={12} color={C.muted}>Key tokens</Text>
        <TokenRow x={137} y={150} n={10} size={25} selected={[0,3,6,8]} tone="rose"/>
        <Path d="M500 127V164H447 M284 132V143" color={C.rose} arrow={false}/>
        <Path d="M243 178V208 M398 178V208" color={C.rose}/>
        <Box x={217} y={214} w={254} h={36} title="Attention on selected K / V" tone="blue"/>
        <Caption x={94} y={236}>top-256</Caption>
        <Particle d="M138 101H199" color={C.blue} duration={2.4}/>
      </>;
      break;
    case 'qsa':
      body=<>
        {[0,1,2].map(b=><g key={b}>
          <TokenRow x={28+b*145} y={82} n={4} size={23} tone="blue"/>
          <Path d={`M${85+b*145} 110V142`} color={C.purple}/>
          <Box x={37+b*145} y={149} w={99} h={43} title={`Block ${b+1}`} sub="pool + norm" tone={b===1?'rose':'purple'} muted={b===2}/>
          {b<2&&<Path d={`M${85+b*145} 194V217`} color={C.rose}/ >}
        </g>)}
        <Box x={474} y={91} w={100} h={64} title="Tail ≤3" sub="always visible" tone="rose"/>
        <Path d="M524 157V239H435" color={C.rose}/>
        <Box x={41} y={222} w={387} h={34} title="Selected complete blocks + visible tail" tone="blue"/>
        <Caption y={278}>Query-dependent block selection · expand selected keys back to tokens</Caption>
      </>;
      break;
    case 'nsa':
      body=<>
        <Box x={22} y={121} w={93} h={53} title="Input" tone="blue"/>
        {[
          {y:84,title:'Compressed',sub:'learned summaries',tone:'purple' as const},
          {y:148,title:'Selected',sub:'important historical blocks',tone:'blue' as const},
          {y:212,title:'Local window',sub:'recent 512 tokens',tone:'rose' as const},
        ].map(b=><g key={b.title}>
          <Path d={`M116 147H140V${b.y}H165`} color={C[b.tone]}/>
          <Box x={172} y={b.y-25} w={204} h={50} title={b.title} sub={b.sub} tone={b.tone}/>
          <Path d={`M378 ${b.y}H405`} color={C[b.tone]}/>
          <circle cx={426} cy={b.y} r={17} fill={C[`${b.tone}Light`]} stroke={C[b.tone]}/>
          <Text x={426} y={b.y+5} anchor="middle" size={17} color={C[b.tone]}>g</Text>
          <Path d={`M444 ${b.y}H473V148H496`} color={C[b.tone]}/>
        </g>)}
        <circle cx={522} cy={148} r={23} fill={C.white} stroke={C.ink} strokeWidth={1.4}/>
        <Text x={522} y={156} anchor="middle" size={27}>Σ</Text><Path d="M548 148H574" color={C.blue}/>
        <Caption y={266}>Compression + selection + local attention · three learned gates</Caption>
      </>;
      break;
    case 'seer':
      body=<>
        <Box x={25} y={101} w={105} h={54} title="Token query" tone="blue"/>
        <Path d="M132 128H177" color={C.blue}/>
        <Box x={184} y={101} w={127} h={54} title="Learned gate" sub="teacher KL" tone="purple"/>
        <Path d="M313 128H344" color={C.purple}/>
        <Text x={449} y={74} size={13} anchor="middle" color={C.purple}>BLOCK SCORES</Text>
        {Array.from({length:24},(_,i)=>{
          const score=[.18,.37,.88,.13,.23,.62,.74,.22,.41,.16,.94,.3,.25,.65,.18,.83,.34,.46,.85,.14,.7,.35,.2,.79][i];
          return <rect key={i} x={357+(i%6)*29} y={86+Math.floor(i/6)*29} width={23} height={23} rx={4}
            fill={C.purple} opacity={score} stroke={score>.7?C.rose:'none'} strokeWidth={3} className={score>.8?'arch-glow':undefined}/>;
        })}
        <Path d="M449 207V229H324" color={C.rose}/>
        <Text x={316} y={234} size={14} anchor="end" color={C.rose}>Select high-score history blocks</Text>
        <Caption y={263}>Seer-style adaptation · token queries / pooled keys · 64-token blocks</Caption>
      </>;
      break;
    case 'moba':
    case 'sparse':
      body=<>
        {[0,1,2,3].map(i=><g key={i}>
          <rect x={35+i*140} y={77} width={109} height={65} rx={10} fill={i===3?C.roseLight:C.blueLight} stroke={i===3?C.rose:C.blue} strokeWidth={1.4}/>
          <Text x={89.5+i*140} y={103} size={15} anchor="middle" weight={650}>{i===3?'Current':'KV block '+(i+1)}</Text>
          <Text x={89.5+i*140} y={124} size={12} anchor="middle" color={C.muted}>{i===3?'causal · always on':'pooled key'}</Text>
        </g>)}
        <Box x={110} y={202} w={120} h={43} title="Query A" tone="purple"/>
        <Box x={366} y={202} w={120} h={43} title="Query B" tone="rose"/>
        <Path d="M151 200L92 149 M195 200L509 149" color={C.purple} flow/>
        <Path d="M402 200L230 149 M451 200L509 149" color={C.rose} dashed/>
        <Caption y={269}>Historical block selection + forced current block · 128-token blocks</Caption>
      </>;
      break;
    case 'deltanet':
      body=<>
        <Box x={24} y={83} w={125} h={56} title="Old state S" sub="matrix memory" tone="blue"/>
        <Path d="M151 110H197" color={C.blue}/>
        <Box x={204} y={83} w={141} h={56} title="Predict kᵀS" sub="read with key" tone="purple"/>
        <Path d="M347 110H389" color={C.purple}/>
        <Box x={396} y={83} w={176} h={56} title="Error: v − kᵀS" sub="what needs correcting" tone="rose"/>
        <Path d="M483 141V158H316V167" color={C.rose}/>
        <Box x={201} y={175} w={230} h={49} title="Write βk · errorᵀ" tone="purple"/>
        <Path d="M86 141V200H193" color={C.blue}/>
        <Path d="M433 200H477" color={C.purple}/>
        <Box x={484} y={175} w={88} h={49} title="S′" sub="qᵀS′ → y" tone="blue"/>
        <Caption y={263}>Delta rule · learned write strength β · normalized keys</Caption>
      </>;
      break;
    case 'gla':
    case 'state':
      body=<>
        <Text x={67} y={76} size={13} anchor="middle" color={C.blue}>Memory S</Text>
        <Matrix x={28} y={100} cell={17}/>
        <Path d="M109 138H144" color={C.blue}/>
        <Text x={174} y={76} size={13} anchor="middle" color={C.purple}>Gate g</Text>
        {[.2,.8,.4,.9].map((a,i)=><g key={i}>
          <rect x={155} y={100+i*20} width={38} height={17} rx={3} fill={C.purple} opacity={a}/>
          <Text x={174} y={112+i*20} size={10} anchor="middle" color={a>.5?C.white:C.purple}>{['g₁','g₂','g₃','g₄'][i]}</Text>
        </g>)}
        <Path d="M196 138H239" color={C.purple} flow/>
        <Text x={289} y={76} size={13} anchor="middle" color={C.purple}>Retained state</Text>
        <Matrix x={250} y={100} cell={17} tone="purple" values={[.1,.18,.12,.15,.6,.74,.55,.68,.22,.33,.3,.25,.9,.71,.84,.65]}/>
        <Text x={174} y={197} size={11} anchor="middle" color={C.muted}>feature-wise decay</Text>
        <Path d="M332 138H359" color={C.purple}/>
        <circle cx={379} cy={138} r={15} fill={C.roseLight} stroke={C.rose}/>
        <Text x={379} y={144} size={20} anchor="middle" color={C.rose}>+</Text>
        <Path d="M396 138H440" color={C.rose}/>
        <Text x={491} y={76} size={13} anchor="middle" color={C.blue}>Updated S′</Text>
        <Matrix x={452} y={100} cell={17} tone="blue"/>
        <Box x={315} y={213} w={130} h={38} title="New write · kvᵀ" tone="rose"/>
        <Path d="M379 211V161" color={C.rose} flow/>
        <Path d="M490 183V203" color={C.blue}/>
        <Text x={492} y={223} size={13} anchor="middle" color={C.blue}>qᵀS′ → output</Text>
        <Caption y={278}>Forget with diag(g) · write a rank-one update · read with the query</Caption>
        <Particle d="M196 138H239" color={C.purple} duration={3}/>
      </>;
      break;
    case 'kda':
      body=<>
        <Text x={81} y={81} size={13} anchor="middle" color={C.blue}>Old state</Text><Matrix x={41} y={95} cell={18}/>
        <Path d="M125 138H165" color={C.blue}/>
        <Text x={204} y={81} size={13} anchor="middle" color={C.purple}>Vector gate</Text>
        {[.2,.75,.4,.92].map((a,i)=><g key={i}><rect x={178} y={95+i*22} width={52} height={18} rx={4} fill={C.purple} opacity={a}/>
          <Text x={204} y={108+i*22} size={10} anchor="middle" color={i===0?C.purple:C.white}>{['g₁','g₂','g₃','g₄'][i]}</Text></g>)}
        <Path d="M234 138H282" color={C.purple} flow/>
        <Text x={329} y={81} size={13} anchor="middle" color={C.purple}>Decayed state</Text>
        <Matrix x={290} y={95} cell={18} tone="purple" values={[.18,.22,.12,.2,.63,.74,.55,.68,.2,.33,.4,.25,.9,.71,.84,.65]}/>
        <Path d="M378 138H416" color={C.rose}/>
        <Box x={423} y={103} w={150} h={68} title="Delta correction" sub="βk(v − kᵀSdecay)ᵀ" tone="rose"/>
        <Path d="M499 173V205H329" color={C.rose}/>
        <Text x={319} y={210} size={15} anchor="end" color={C.rose}>Write a corrected memory</Text>
        <Caption y={259}>Per-feature decay, rather than one scalar gate for the entire head</Caption>
      </>;
      break;
    case 'retnet': {
      const curves=[{rate:5,color:C.rose,name:'Fast decay'},{rate:2,color:C.purple,name:'Medium decay'},{rate:.45,color:C.blue,name:'Slow decay'}];
      body=<>
        <Path d="M64 211H562 M64 211V75" color={C.line} arrow={false} width={1.2}/>
        {[0,.5,1].map((v,i)=><g key={i}><line x1={64} x2={560} y1={211-v*124} y2={211-v*124} stroke="#E6ECF2" strokeDasharray="3 5"/>
          <Text x={51} y={215-v*124} size={11} anchor="end" color={C.muted}>{v}</Text></g>)}
        {curves.map((c,i)=>{const d=Array.from({length:50},(_,n)=>`${n?'L':'M'}${64+n*10} ${211-Math.exp(-c.rate*n/49)*124}`).join(' ');return <g key={c.name}>
          <path d={d} fill="none" stroke={c.color} strokeWidth={3} strokeLinecap="round"/>
          <Particle d={d} color={c.color} delay={i*.6} duration={5}/>
          <circle cx={109+i*177} cy={246} r={4} fill={c.color}/><Text x={121+i*177} y={251} size={13} color={c.color}>{c.name}</Text>
        </g>;})}
        <Text x={304} y={230} size={12} anchor="middle" color={C.muted}>Distance into the past →</Text>
        <Text x={82} y={76} size={11} color={C.muted}>retention weight</Text>
      </>;
      break;
    }
    case 'gdn':
      body=<>
        <Text x={81} y={81} size={13} anchor="middle" color={C.blue}>Old state</Text><Matrix x={41} y={95} cell={18}/>
        <Path d="M125 138H165" color={C.blue}/>
        <Text x={204} y={81} size={13} anchor="middle" color={C.purple}>Head gate α</Text>
        <rect x={178} y={95} width={52} height={84} rx={6} fill={C.purple} opacity={.62}/>
        <Text x={204} y={142} size={13} anchor="middle" color={C.white}>α</Text>
        <Path d="M234 138H282" color={C.purple} flow/>
        <Text x={329} y={81} size={13} anchor="middle" color={C.purple}>Decayed state</Text>
        <Matrix x={290} y={95} cell={18} tone="purple" values={[.38,.42,.35,.4,.38,.44,.36,.41,.39,.43,.37,.4,.42,.36,.4,.38]}/>
        <Path d="M378 138H416" color={C.rose}/>
        <Box x={423} y={103} w={150} h={68} title="Delta correction" sub="βk(v − kᵀαS)ᵀ" tone="rose"/>
        <Path d="M499 173V205H329" color={C.rose}/>
        <Text x={319} y={210} size={15} anchor="end" color={C.rose}>Write a corrected memory</Text>
        <Caption y={259}>One scalar decay per head and token (Mamba2 form), rather than per feature</Caption>
      </>;
      break;
    case 'elu_qknorm':
      body=<>
        <Box x={18} y={115} w={128} h={60} title="Fixed q/k scale" sub="RMSNorm(q, k) × 8" tone="purple"/>
        <Path d="M148 145H170" color={C.purple}/>
        <Box x={176} y={115} w={122} h={60} title="Positive features" sub="φ = ELU + 1" tone="blue"/>
        <Path d="M300 145H322" color={C.blue}/>
        <Box x={328} y={115} w={118} h={60} title="S accumulator" sub="S += φ(k)vᵀ" tone="purple"/>
        <Path d="M448 145H470" color={C.purple}/>
        <Box x={476} y={115} w={100} h={60} title="Per-head norm" sub="RMSNorm · gain" tone="rose"/>
        <Caption y={259}>y = RMSNorm(φ(q)ᵀS) · gain — no denominator; IEEE FP32 attention</Caption>
      </>;
      break;
    case 'mqa':
    case 'gqa': {
      const groups = methodId === 'mqa' ? 1 : 3;
      const qx = (i: number) => 30 + i * 46;
      const center = (g: number) => groups === 1 ? 301 : qx(g * 4) + 87;
      body = <>
        {Array.from({ length: 12 }, (_, i) => <g key={i}>
          <rect x={qx(i)} y={70} width={36} height={40} rx={7} fill={C.blueLight} stroke={C.blue} strokeWidth={1.2}/>
          <Text x={qx(i)+18} y={95} size={12} anchor="middle" weight={600}>{`q${i+1}`}</Text>
          <Path d={`M${qx(i)+18} 112L${center(Math.floor(i * groups / 12))} 166`} color={C.purple} arrow={false} width={1.4} opacity={.6}/>
        </g>)}
        {Array.from({ length: groups }, (_, g) => <Box key={g} x={center(g) - (groups === 1 ? 100 : 75)} y={168} w={groups === 1 ? 200 : 150} h={56}
          title={groups === 1 ? 'Shared key, value' : `Key, value · ${g + 1}`} sub={groups === 1 ? 'one head · dimension 128' : 'four query heads'} tone="purple"/>)}
        <Caption y={252}>{groups === 1 ? 'Twelve query heads read one key/value head' : 'Twelve query heads in three groups of four'}</Caption>
        <Caption y={274}>{groups === 1 ? 'RoPE and QK RMSNorm × 1.2 act on the shared key once' : 'RoPE and QK RMSNorm × 1.2 act on each key head once'} · no value embeddings</Caption>
      </>;
      break;
    }
    case 'mla_qknorm':
    case 'mla_nope_qknorm': {
      const rope = methodId === 'mla_qknorm';
      body = <>
        <Box x={8} y={150} w={78} h={56} title="Hidden x" sub="1536" tone="blue"/>
        <Path d="M88 178H114" color={C.purple}/>
        <Path d="M88 178H96V67H114" color={C.blue}/>
        <Box x={118} y={42} w={110} h={50} title="Query latent" sub="rank 768" tone="blue"/>
        <Path d="M230 67H254" color={C.blue}/>
        <Box x={258} y={42} w={146} h={50} title="Queries × 12" sub={rope ? 'content 64 + RoPE 64' : 'content 128'} tone="blue"/>
        <Box x={118} y={150} w={110} h={56} title="KV latent" sub="rank 256" tone="purple"/>
        <Path d="M230 178H242V167H254" color={C.purple}/>
        <Box x={258} y={142} w={146} h={50} title="Keys × 12" sub={rope ? 'content 64 + RoPE 64' : 'content 128'} tone="purple"/>
        <Path d="M230 178H242V250H254" color={C.purple}/>
        <Box x={258} y={225} w={146} h={50} title="Values × 12" sub="128 per head" tone="purple"/>
        {rope ? <>
          <Path d="M96 178V118H331V138" color={C.rose}/>
          <Text x={240} y={111} size={12} anchor="middle" color={C.rose}>Shared rotary key · 64 · from x</Text>
        </> : <Text x={240} y={120} size={12} anchor="middle" color={C.muted}>No rotary query or key channels</Text>}
        <Path d="M406 67H420V99H434" color={C.blue}/>
        <Path d="M406 167H420V140H434" color={C.purple}/>
        <Box x={438} y={82} w={148} h={76} title="QK RMSNorm ×1.2" sub="queries and keys only" tone="rose"/>
        <Path d="M512 160V214" color={C.rose}/>
        <Text x={526} y={190} size={11} color={C.rose}>Q, K</Text>
        <Path d="M406 250H434" color={C.purple}/>
        <Box x={438} y={218} w={148} h={65} title="Causal attention" sub="softmax(QKᵀ / √128)V" tone="blue"/>
        <Path d="M512 285V302" color={C.blue}/>
        <Text x={512} y={318} size={13} anchor="middle" color={C.blue}>Output projection</Text>
        <Text x={190} y={300} size={12} color={C.purple}>Values bypass QK normalization</Text>
        <Caption y={343}>12 heads · each logit ≤ 1.44·√128 ≈ 16.3 · no value embeddings</Caption>
      </>;
      break;
    }
    case 'residual_standard':
      body = <>
        <rect x={107} y={77} width={400} height={119} rx={16} fill="#FBF9FE" stroke="#D6C7E9" strokeDasharray="5 5"/>
        <Text x={132} y={98} size={12} color={C.purple}>EVERY ATTENTION AND FFN SUBLAYER</Text>
        <Text x={27} y={151} size={18} weight={700}>h</Text>
        <Path d="M44 145H127" color={C.purple}/>
        <Box x={134} y={118} w={101} h={52} title="RMSNorm" tone="purple"/>
        <Path d="M237 144H262" color={C.purple}/>
        <Box x={269} y={118} w={120} h={52} title="F" sub="attention or FFN" tone="purple"/>
        <Path d="M391 144H416" color={C.purple}/>
        <circle cx={441} cy={144} r={18} fill={C.roseLight} stroke={C.rose}/><Text x={441} y={150} size={21} anchor="middle" color={C.rose}>+</Text>
        <Path d="M63 144V62H441V118" color={C.rose}/>
        <Text x={252} y={56} size={13} color={C.rose} anchor="middle">identity · no learned residual or x0 scale</Text>
        <Path d="M461 144H526" color={C.purple}/>
        <Text x={533} y={151} size={18} weight={700}>h′</Text>
        <Path d="M441 199C441 231 178 231 149 200" color={C.purple} flow/>
        <Text x={295} y={246} size={13} anchor="middle" color={C.purple}>46 sublayers · 23 layers · FFN 6144</Text>
        <Caption y={272}>Native residual / x0 scalars and backout stay frozen and unused</Caption>
      </>;
      break;
    case 'attnres': {
      const sources: [string, string][] = [['Embedding', 'h₀'], ['Group 1', 'sum'], ['Group 2', 'sum'], ['…', ''], ['Group g−1', 'sum'], ['Partial g', 'current group']];
      body = <>
        {sources.map(([title, sub], i) => <g key={i}>
          <Box x={14+i*98} y={58} w={84} h={46} title={title} sub={sub || undefined} tone={i === 0 ? 'blue' : 'purple'} muted={i === 3}/>
          <Path d={`M${56+i*98} 106L300 138`} color={C.purple} arrow={false} width={1.3} opacity={.55}/>
        </g>)}
        <Box x={200} y={140} w={200} h={50} title="Softmax over depth" sub="RMSNorm keys" tone="purple"/>
        <Box x={24} y={142} w={142} h={46} title="Pseudo-query" sub="per sublayer · zero init" tone="rose"/>
        <Path d="M168 165H194" color={C.rose}/>
        <Path d="M300 192V204" color={C.purple}/>
        <Box x={210} y={208} w={180} h={42} title="Attention or FFN" tone="purple"/>
        <Path d="M392 229H546V108" color={C.purple} flow/>
        <Text x={470} y={221} size={12} anchor="middle" color={C.purple}>joins the partial sum</Text>
        <Caption y={276}>8 groups (3,3,3,3,3,3,3,2 blocks) · the readout attends over all eight group sums</Caption>
      </>;
      break;
    }
    case 'gsa_qknorm':
      viewBox = '0 80 600 166';
      body = <>
        <Text x={514} y={104} size={14} anchor="end" weight={600} color={C.purple}>v</Text>
        <Path d="M524 86V110" color={C.purple}/>
        <Box x={20} y={116} w={130} h={56} title="q, k" sub="QK-norm · SiLU" tone="blue"/>
        <Path d="M152 144H170" color={C.purple}/>
        <Box x={176} y={116} w={130} h={56} title="Key slots" sub="64 slots · gated" tone="purple"/>
        <Path d="M308 144H326" color={C.purple}/>
        <Box x={332} y={116} w={110} h={56} title="Slot softmax" sub="query → slots" tone="rose"/>
        <Path d="M444 144H462" color={C.purple}/>
        <Box x={468} y={116} w={112} h={56} title="Value slots" sub="slots → output" tone="purple"/>
        <Path d="M241 174V190H524V174" color={C.purple} arrow={false} dashed/>
        <Text x={382} y={208} size={12} anchor="middle" color={C.purple}>one per-slot gate: decay α, write 1 − α</Text>
        <Caption y={238}>24 layers · two slot memories, read with a softmax over 64 slots</Caption>
      </>;
      break;
    case 'gdn_hybrid':
      body = <>
        <rect x={40} y={78} width={530} height={118} rx={16} fill="#FBF9FE" stroke="#D6C7E9" strokeDasharray="5 5"/>
        <Text x={62} y={100} size={12} color={C.purple}>ONE GROUP · REPEATED SIX TIMES</Text>
        {['GDN','GDN','GDN','MHA'].map((title,i)=><g key={i}>
          <Box x={62+i*126} y={116} w={104} h={58} title={title} sub={i<3?'gated delta':'full causal'} tone={i<3?'blue':'purple'}/>
          {i<3&&<Path d={`M${168+i*126} 145H${182+i*126}`} color={i<2?C.blue:C.purple}/>}
        </g>)}
        <Path d="M492 199C492 236 114 236 114 199" color={C.purple} flow/>
        <Text x={303} y={252} size={13} anchor="middle" color={C.purple}>× 6 · 18 GDN + 6 MHA layers · independent weights</Text>
        <Caption y={276}>Width 1536 · FFN 5760 · GDN state resets at each packed row</Caption>
      </>;
      break;
    case 'gdn_hybrid_swa':
      body = <>
        <rect x={40} y={78} width={530} height={118} rx={16} fill="#FBF9FE" stroke="#D6C7E9" strokeDasharray="5 5"/>
        <Text x={62} y={100} size={12} color={C.purple}>ONE PAIR · REPEATED ELEVEN TIMES</Text>
        <Box x={110} y={116} w={160} h={58} title="GDN" sub="gated delta" tone="blue"/>
        <Path d="M272 145H334" color={C.purple}/>
        <Box x={340} y={116} w={160} h={58} title="SWA" sub="window 2048 = full 2K" tone="purple"/>
        <Path d="M420 199C420 236 190 236 190 199" color={C.purple} flow/>
        <Text x={305} y={252} size={13} anchor="middle" color={C.purple}>× 11 · 11 GDN + 11 SWA layers · independent weights</Text>
        <Caption y={276}>Width 1536 · FFN 6528 · GDN state resets at each packed row</Caption>
      </>;
      break;
    default: {
      const labels=(nodes?.length?nodes:['Input','Mixer','MLP','Output']).filter(Boolean).slice(0,8);
      const cols=labels.length<=4?labels.length:Math.ceil(labels.length/2), rows=Math.ceil(labels.length/cols);
      const gap=24, width=nodeWidth(cols), left=(600-cols*width-(cols-1)*gap)/2;
      const {size,lines}=fitLabels(labels,width-10), lead=size*1.25;
      const h=lines.some(l=>l.length>1)?58:52, top=60, bottom=top+(rows-1)*(h+40)+h, loopY=bottom+24;
      const captionY=(loop&&labels.length>1?loopY:bottom)+40;
      viewBox=`0 ${top-24} 600 ${captionY+16-(top-24)}`;
      const pos=labels.map((name,i)=>({name,lines:lines[i],x:left+(i%cols)*(width+gap),y:top+Math.floor(i/cols)*(h+40)}));
      body=<>
        {pos.map((p,i)=>{const tone=i===0?'blue':i===labels.length-1?'rose':'purple', next=pos[i+1];
          return <g key={i}>
            <rect x={p.x} y={p.y} width={width} height={h} rx={7} fill={C[`${tone}Light`]} stroke={C[tone]} strokeWidth={1.2}><title>{p.name}</title></rect>
            <Text x={p.x+width/2} y={p.y+h/2+size*.36-(p.lines.length-1)*lead/2} size={size} anchor="middle" weight={600}>
              {p.lines.map((line,j)=><tspan key={j} x={p.x+width/2} dy={j?lead:0}>{line}</tspan>)}
            </Text>
            {next&&<Path d={next.y===p.y?`M${p.x+width+3} ${p.y+h/2}H${next.x-6}`:`M${p.x+width/2} ${p.y+h+3}V${p.y+h+20}H${next.x+width/2}V${next.y-6}`} color={C.purple}/>}
          </g>;})}
        {loop&&pos.length>1&&<Path d={`M${pos.at(-1)!.x+width/2} ${pos.at(-1)!.y+h+3}V${loopY}H${pos[0].x+width/2}V${pos[0].y+h+6}`} color={C.purple} flow/>}
        <Caption y={captionY}>{loop?'A recurring path is shown; exact sharing follows the submitted implementation.':'One-way flow through the submitted nodes.'}</Caption>
      </>;
    }
  }

  return <figure className="architecture-figure"><figcaption><strong>{description[0]}</strong><span>{description[1]}</span></figcaption><div className="architecture-canvas"><svg viewBox={viewBox} className={`architecture-diagram${large?' architecture-large':''}`}
    style={{display:'block',width:'100%',height:'auto',overflow:'hidden',fontFamily:'Inter, ui-sans-serif, system-ui, -apple-system, sans-serif'} as CSSProperties}
    role="img" aria-label={`${label} — illustrative model mechanism`}>
    <defs><marker id={arrowId} markerWidth="7" markerHeight="7" refX="5.2" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M1 1L5.5 3.5L1 6" fill="none" stroke="context-stroke" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/>
    </marker></defs>
    <style>{`
      .architecture-diagram .arch-flow { animation: arch-flow 3s linear infinite; }
      .architecture-diagram .arch-glow { animation: arch-glow 3.4s ease-in-out infinite; }
      @keyframes arch-flow { to { stroke-dashoffset: -56; } }
      @keyframes arch-glow { 50% { opacity: .5; } }
      @media (prefers-reduced-motion: reduce) {
        .architecture-diagram .arch-flow, .architecture-diagram .arch-glow { animation: none !important; }
        .architecture-diagram .arch-moving { display: none; }
      }
    `}</style>
    {body}
  </svg></div></figure>;
}
