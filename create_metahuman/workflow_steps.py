"""Display order with stable numbers for projects saved before 0.27."""
STAGES = (
    ('START', '1. Source & Template', 0),
    ('WRAP', '2. Wrap', 1),
    ('REFINE', '3. Refine', 2),
    ('TEXTURE', '4. Texture', 3),
    ('DETAIL', '5. Detail', 4),
    ('EXPORT', '6. Export', 6),
    ('GROOM', '7. Groom (Optional)', 5),
)
ORDER = tuple(stage for stage, _, _ in STAGES)
ITEMS = [(stage, label, '', 0, number) for stage, label, number in STAGES]
