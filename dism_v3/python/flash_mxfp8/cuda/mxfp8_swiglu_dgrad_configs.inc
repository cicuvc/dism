// Fused dX = dG @ Wg + dU @ Wu.
// CFG(id, per-warp output rows, output columns, reduction tile, stages)
CFG(0,16,32,32,2);
CFG(1,16,64,32,2);
CFG(2,16,128,32,2);
CFG(3,32,32,32,2);
CFG(4,32,64,32,2);
CFG(5,16,64,64,2);
CFG(6,16,128,64,2);
CFG(7,32,32,64,2);
CFG(8,16,32,128,2);
CFG(9,32,64,32,3);
CFG(10,16,64,128,1);
CFG(11,16,128,128,1);
CFG(12,16,64,64,3);
CFG(13,16,64,32,3);
CFG(14,16,32,64,3);
