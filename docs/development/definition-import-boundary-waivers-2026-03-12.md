# Definition Import Boundary Waiver Process (2026-03-12)

Purpose: define the `VPC-T5.3` exception process for definition-layer import
guardrails so temporary migration waivers stay explicit, reviewable, and
time-bounded.

## Default Rule

Definition-layer modules must not import runtime-owned Victor modules.

Forbidden prefixes are enforced in
`tests/unit/core/verticals/test_definition_import_boundaries.py`:

- `victor.framework`
- `victor.core.verticals`
- `victor.tools.tool_names`
- `victor.framework.tool_naming`

Allowed package-local imports are limited to other definition-layer helpers such
as `prompt_metadata.py`.

## What Counts As A Waiver

A waiver is any temporary allowance that would let a definition-layer file depend
on a Victor runtime-owned module or import path that would normally fail the
guardrail.

Current status:

- no runtime-import waivers are approved
- existing allowed local imports in the test file are not waivers; they are
  definition-layer helper imports

## Waiver Requirements

Any future waiver must include all of the following in the guardrail test file:

1. exact file or target label covered by the waiver
2. exact import prefix being allowed temporarily
3. linked roadmap task or issue
4. owner role
5. expiry release or expiry date
6. short rationale for why the migration cannot be completed immediately

If any of that information is missing, the waiver should not be added.

## Non-Negotiable Restrictions

The following are not eligible for open-ended waivers:

- `victor.framework`
- `victor.core.verticals`
- `victor.tools.tool_names`
- `victor.framework.tool_naming`

If one of those prefixes must be tolerated temporarily, the waiver must be
time-bounded and tied to a specific migration task. Permanent exceptions are not
allowed.

## Review And Removal Rules

- every waiver must be reviewed in code review
- every waiver must be called out in the roadmap session log
- every waiver must be removed as part of the linked migration task
- releases carrying a waiver must mention it in migration/deprecation notes if it
  affects external authors

## Local Workflow

Maintain the guardrail locally with:

```bash
make test-definition-boundaries
```

This command should stay fast enough to run before pushing any definition-layer
changes.
