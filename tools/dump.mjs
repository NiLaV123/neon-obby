import * as S from '../web/shaders.js';
import fs from 'fs';
for (const [k, v] of Object.entries(S)) fs.writeFileSync(`${process.argv[2]}/${k}.wgsl`, v);
console.log(Object.keys(S).join(' '));
