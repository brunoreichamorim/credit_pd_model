-- Illustrative risk grade of every scored loan (decisions D-014, D-027): a range join of
-- pd_scores to the grade_scale table built by src/grades.py. Each grade covers
-- pd_lower <= pd < pd_upper; the top grade also includes its upper bound (1.0), so every
-- scored loan gets exactly one grade (tests/test_grades.py checks this).
-- Only the PD is used: no outcome is read here. EQUI loans are not in pd_scores (D-026),
-- so they get no grade.
CREATE OR REPLACE VIEW pd_grades AS
SELECT
    s.ID,
    s.sample,
    s.pd,
    g.grade,
    g.grade_rank,
    g.grade_pd
FROM pd_scores AS s
JOIN grade_scale AS g
    ON s.pd >= g.pd_lower
   AND (s.pd < g.pd_upper OR g.pd_upper = 1.0);
