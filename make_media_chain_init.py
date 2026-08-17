#!/usr/bin/env python
"""Emit the media-chain initialization table used by get_V0's V0_init=-2 mode.

For every media compound the exchange, the e0<->c0 transporter and the c0<->d0
transporter form a series chain: at steady state all three carry the same flux.
So they are all initialized to the same value -- the exchange's net FVA capacity
-- on the column matching the exchange's own direction, with the opposing column
set to 0.

Writes <project>/integration_results/media_chain_init.tsv
"""
import glob, os, sys
import xml.etree.ElementTree as ET

NS = {'s': 'http://www.sbml.org/sbml/level3/version1/core'}

def compartment(species_id):
    for c in ('e0', 'c0', 'd0', 'y0'):
        if species_id.endswith('_' + c):
            return c
    return '?'

def build(project_folder):
    xml = glob.glob(os.path.join(project_folder, "inputs", "*_dup.xml"))[0]
    model = ET.parse(xml).getroot().find('s:model', NS)

    reactants, products = {}, {}
    for rx in model.find('s:listOfReactions', NS).findall('s:reaction', NS):
        rid = rx.get('id').replace('R_', '')
        reactants[rid] = [x.get('species').replace('M_', '')
                          for x in rx.findall('s:listOfReactants/s:speciesReference', NS)]
        products[rid] = [x.get('species').replace('M_', '')
                         for x in rx.findall('s:listOfProducts/s:speciesReference', NS)]

    # exchange net capacity == raw FVA max (exchanges have no directional partner)
    fva = {}
    with open(os.path.join(project_folder, "integration_results", "fva.tsv")) as fh:
        fh.readline()
        for line in fh:
            rxn, mx = line.rstrip('\n').split('\t')
            fva[rxn] = abs(float(mx))

    # index every reaction by the species it touches
    touches = {}
    for rid in reactants:
        for m in reactants[rid] + products[rid]:
            touches.setdefault(m, []).append(rid)

    rows = []
    for ex in sorted(r for r in reactants if r.startswith('EX_')):
        value = fva.get(ex, 0.0)
        direction = ex.rsplit('_', 1)[-1]              # 'i' (import) or 'o' (export)
        opposite = 'o' if direction == 'i' else 'i'
        met_e0 = (reactants[ex] + products[ex])[0]
        base_cpd = met_e0.rsplit('_', 1)[0]
        rows.append((ex, value, base_cpd, 'exchange'))

        # walk inward: e0 -> next compartment -> plastid
        frontier = [met_e0]
        seen_rxn = {ex}
        seen_met = {met_e0}
        while frontier:
            met = frontier.pop()
            for rid in touches.get(met, []):
                if rid in seen_rxn or rid.startswith('EX_') or rid == 'bio1':
                    continue
                partners = [m for m in reactants[rid] + products[rid]
                            if m.rsplit('_', 1)[0] == base_cpd and compartment(m) != compartment(met)]
                if not partners:
                    continue                            # not a transport step for this cargo
                stem, sfx = rid.rsplit('_', 1)
                if sfx not in ('i', 'o', 'f', 'r'):
                    continue                            # unsplit (e.g. thylakoid pumps): leave alone
                seen_rxn.add(rid)
                seen_rxn.add(f"{stem}_{opposite if sfx == direction else direction}")
                rows.append((f"{stem}_{direction}", value, base_cpd, 'transport'))
                rows.append((f"{stem}_{opposite}", 0.0, base_cpd, 'transport-opposing'))
                for m in partners:
                    if m not in seen_met:
                        seen_met.add(m)
                        frontier.append(m)

    out = os.path.join(project_folder, "integration_results", "media_chain_init.tsv")
    with open(out, 'w') as fh:
        fh.write("reaction\tvalue\tcompound\trole\n")
        for rid, val, cpd, role in rows:
            fh.write(f"{rid}\t{val:.6f}\t{cpd}\t{role}\n")
    print(f"wrote {out}  ({len(rows)} columns)")
    return rows

if __name__ == '__main__':
    for pf in sys.argv[1:]:
        for rid, val, cpd, role in build(pf):
            if role != 'transport-opposing':
                print(f"    {rid:22s} {val:10.3f}  {cpd}  {role}")
