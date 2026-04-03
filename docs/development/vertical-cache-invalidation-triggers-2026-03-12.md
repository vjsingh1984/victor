# Vertical Cache Invalidation Triggers (2026-03-12)

Purpose: define `VPC-T5.9` and anchor the first runtime invalidation hook for
`VPC-T5.10` so package-changing flows have an explicit host-owned refresh
contract instead of relying on TTL expiry or manual test-only resets.

## Scope

- Included: runtime discovery/config caches relevant to vertical package changes
- Included: install, uninstall, upgrade, and reload triggers
- Included: the current host-owned invalidation path wired through
  `VerticalRegistryManager`
- Excluded: future CLI-level explicit refresh UX (`VPC-T5.12`)
- Excluded: resolver simplification and packaging flips (`VPC-E4`)

## Current Cache Surfaces

The current vertical package/runtime path caches state in several places:

| Surface | Current owner | Why it matters after package changes |
|---|---|---|
| `VerticalBase` config + extension caches | `victor.core.verticals.base.VerticalBase` | A vertical upgrade can leave stale config/extension payloads in memory |
| Registry discovery flag and registry state | `VerticalRegistry` | `discover_external_verticals()` short-circuits after first discovery |
| `VerticalLoader` discovered vertical/tool caches | `victor.core.verticals.vertical_loader.VerticalLoader` | Loader can keep stale entry-point results for the life of the process |
| Entry-point cache | `victor.framework.module_loader.EntryPointCache` | Installed/uninstalled packages change entry-point membership |
| Framework entry-point helper cache | `victor.framework.entry_point_loader` | Runtime add-on entry points can drift after installs/upgrades |
| Tool dependency provider caches | `victor.core.tool_dependency_loader` | Tool dependency resolution can point at stale provider modules |

## Trigger Matrix

| Trigger | When it happens | Required invalidation action | Current implementation status |
|---|---|---|---|
| Install | A new vertical package is successfully installed | Clear config caches, reset discovery, refresh loader/plugin caches | Implemented |
| Uninstall | An external vertical package is successfully removed | Clear config caches, reset discovery, refresh loader/plugin caches | Implemented |
| Upgrade | A package is reinstalled at a new version through `pip install` or equivalent | Same as install, because the current runtime cannot trust version-local caches | Implemented; install flow classifies fresh install vs upgrade before invalidation |
| Reload | Operator/test explicitly requests fresh package discovery without package mutation | Clear config caches, reset discovery, refresh loader/plugin caches | Defined in helper; explicit CLI UX remains follow-on work |

## Design Decision

Use one host-owned invalidation helper for package-topology changes:

- `victor.core.verticals.cache_invalidation.invalidate_vertical_runtime_state(...)`

Responsibilities:

1. clear all `VerticalBase` config/extension caches
2. reset `VerticalRegistry` discovery state
3. refresh `VerticalLoader` plugin caches, which also clears:
   - entry-point cache groups
   - extension loader cache
   - framework entry-point loader cache
   - tool dependency entry-point/provider caches

This is intentionally host-owned so extracted vertical packages do not need to
know how the runtime cache stack works.

## Why TTL Alone Is Not Enough

The existing config TTL helps with repeated reads but does not cover:

- new entry points appearing after install
- removed entry points disappearing after uninstall
- runtime add-on entry points changing after upgrade
- process-local discovery flags short-circuiting rediscovery

Package changes therefore need explicit invalidation, not just timed expiry.

## Current Hook Points

Current implementation for `VPC-T5.10`:

- `VerticalRegistryManager.install(...)` invalidates runtime caches after a
  successful `pip install`, classifying the change as `install` vs `upgrade`
  before invalidation
- `VerticalRegistryManager.uninstall(...)` invalidates runtime caches after a
  successful `pip uninstall`

Current implementation for `VPC-T5.12`:

- `victor vertical install ...` and `victor vertical uninstall ...` now tell the
  user that the current process refreshed package caches and only other Victor
  sessions still require restart

This keeps the first refresh hook close to the package-changing operations
already exposed through the Victor CLI.

## Follow-On Work Unblocked

### `VPC-T5.11`

- add tests for cache invalidation semantics beyond the install/uninstall happy
  path
- verify TTL bypass and explicit refresh behavior

### `VPC-T5.12`

- add CLI-level tests and possibly an explicit refresh/reload command or
  improved install/uninstall messaging

## Summary Decision

Install, uninstall, upgrade, and explicit reload are all cache-invalidation
triggers. The runtime now has one shared host-owned invalidation path, and the
package manager install/uninstall flow uses it immediately after successful
package changes.
