import * as S from '../web/shaders.js';
import fs from 'fs';
for (const k of ['RASTER_WGSL','TRACE_WGSL','SHADE_WGSL','POST_WGSL']) fs.writeFileSync(`${process.argv[2]}/${k}.wgsl`, S[k]);
