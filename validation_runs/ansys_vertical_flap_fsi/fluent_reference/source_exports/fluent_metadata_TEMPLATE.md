# Fluent Metadata Template

Use this template for a future provenance-backed ANSYS Fluent export. Do not rename this file to the active metadata filename until every field is populated from a real Fluent run.

- Source document:
  - Required. Cite the exact Fluent case document or internal run record.
- Fluent run id:
  - Required. Use a stable run identifier, not `MISSING`.
- Export author:
  - Required. Name the person or automation that exported the data.
- Export date:
  - Required. Use ISO date format.
- Fluent version:
  - Required. Record the exact ANSYS Fluent version.
- mesh/domain source:
  - Required. Describe the mesh and modeled domain source.
- geometry units:
  - Required. Use the units used by the exported Fluent reports.
- material model:
  - Required. Record fluid and solid material assumptions.
- boundary conditions:
  - Required. Include inlet, outlet, wall/flap, and symmetry assumptions.
- time step:
  - Required. Must match `0.0005` for the current contract.
- number of steps:
  - Required. Must match `50` for the current contract.
- coupling settings if applicable:
  - Required. Describe the Fluent/System Coupling settings used.
- export procedure:
  - Required. Describe exactly how each CSV report was exported.
- who/when/how generated:
  - Required. Record reproducibility details for the export.
- force_z_positive:
  - Required. Define positive force direction.
- flow_rate_positive:
  - Required. Define positive flow-rate direction.
- pressure_reference:
  - Required. Define the pressure reference convention.
- displacement_definition:
  - Required. Define the point/location and displacement norm used.
