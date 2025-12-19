% ---------- Domain ----------
idx(1..5).
posdom(0..4).

% ---------- Input facts (e.g.) ----------
% val(1,X1). val(2,X2). ... val(5,X5).

% ---------- Exactly one position per element ----------
1 { pos(I,P) : posdom(P) } 1 :- idx(I).

% ---------- All-different positions (Permutation) ----------
:- pos(I,P), pos(J,P), I < J.

% ---------- Ordering constraints (stable tie-break) ----------
% If Vi < Vj then Pi < Pj
:- val(I,Vi), val(J,Vj), Vi < Vj,
   pos(I,Pi), pos(J,Pj),
   Pi >= Pj.

% If Vi > Vj then Pi > Pj  (optional, but makes intent explicit)
:- val(I,Vi), val(J,Vj), Vi > Vj,
   pos(I,Pi), pos(J,Pj),
   Pi <= Pj.

% Tie-break: if Vi == Vj and I < J then Pi < Pj
:- val(I,V), val(J,V), I < J,
   pos(I,Pi), pos(J,Pj),
   Pi >= Pj.
