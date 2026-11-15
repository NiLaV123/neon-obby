import fs from 'fs';
const { instance } = await WebAssembly.instantiate(fs.readFileSync(process.argv[2]), {});
const w = instance.exports;
w.init();
const out = () => new Float32Array(w.memory.buffer, w.out_ptr(), 32);
const step = (n, mx, mz, jump) => { for (let i = 0; i < n; i++) w.frame(1/60, mx, mz, jump, Math.PI, 0.3, 18); return out(); };
let o = step(60, 0, 0, 0);
console.log('idle pos', [...o.slice(6, 9)].map(v=>v.toFixed(2)), 'grounded', o[15], 'eye', [...o.slice(0,3)].map(v=>v.toFixed(1)));
console.log('boxes', w.box_count(), 'nodes', w.node_count(), 'prims', w.prim_count(), 'stages', o[10]);
o = step(30, 0, 1, 0); console.log('walk fwd 0.5s z', o[8].toFixed(2), 'facing', o[16].toFixed(2));
o = step(1, 0, 0, 1); o = step(10, 0, 0, 0); console.log('jump mid y', o[7].toFixed(2));
o = step(40, 0, 0, 0); console.log('landed y', o[7].toFixed(2), 'g', o[15]);
// walk off edge into lava -> death expected
o = step(240, 0, 1, 0); console.log('after walking off: deaths', o[11], 'pos', [...o.slice(6,9)].map(v=>v.toFixed(1)));
// BVH sanity: every prim referenced once
const nodes = new Uint32Array(w.memory.buffer, w.nodes_ptr(), w.node_count()*8);
const prims = new Uint32Array(w.memory.buffer, w.prims_ptr(), w.prim_count());
const seen = new Set(); const stack=[0]; while (stack.length){ const i=stack.pop(); const lf=nodes[i*8+3], c=nodes[i*8+7]; if(c>0){for(let k=0;k<c;k++)seen.add(prims[lf+k]);} else stack.push(lf, lf+1);}
console.log('bvh covers', seen.size, 'of', w.box_count());
let t=performance.now(); step(600,0,1,0); console.log('ms/frame', ((performance.now()-t)/600).toFixed(3));
