import numpy as np
from pathlib import Path

from esmda4d.model import (
    GridProperty,
    build_model_ensemble,
)
from esmda4d.cmg.writer import CMGModelWriter


# --------------------------------------------------
# Output location
# --------------------------------------------------

output_dir = Path("example_output")


# --------------------------------------------------
# Create a small CMG template
# --------------------------------------------------

template_path = Path("examples/model.tpl")

template_path.write_text(
    "*POR ALL\n"
    "INCLUDE '$$porosity'\n"
    "\n"
    "*PERMI ALL\n"
    "INCLUDE '$$permeability'\n"
    "\n"
    "*NETGROSS ALL\n"
    "INCLUDE '$$ntg'\n"
)


# --------------------------------------------------
# Create 3 grid cells and 2 realizations
# --------------------------------------------------

cell_ids = np.array([1, 2, 3])

porosity = GridProperty(
    variable="porosity",
    cell_ids=cell_ids,
    values=np.array([
        [0.10, 0.11],
        [0.20, 0.21],
        [0.30, 0.31],
    ]),
)

permeability = GridProperty(
    variable="permeability",
    cell_ids=cell_ids,
    values=np.array([
        [100.0, 110.0],
        [200.0, 210.0],
        [300.0, 310.0],
    ]),
)

ntg = GridProperty(
    variable="ntg",
    cell_ids=cell_ids,
    values=np.array([
        [0.80, 0.81],
        [0.90, 0.91],
        [1.00, 0.99],
    ]),
)


# --------------------------------------------------
# Build ES-MDA model matrix
# --------------------------------------------------

M, metadata = build_model_ensemble([
    porosity,
    permeability,
    ntg,
])

print("M:")
print(M)

print("\nM shape:")
print(M.shape)


# --------------------------------------------------
# Create CMG files
# --------------------------------------------------

writer = CMGModelWriter(
    template_path=template_path,
    output_dir=output_dir,
)

model_paths = writer.write_ensemble(
    M=M,
    metadata=metadata,
)


# --------------------------------------------------
# Show generated files
# --------------------------------------------------

print("\nGenerated CMG models:")

for path in model_paths:
    print(path)