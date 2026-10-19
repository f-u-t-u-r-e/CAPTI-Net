"""Fixed manuscript pathway combinations and first-order interaction controls."""
VARIANTS = {
    'full': (.5, True, True, True, True),
    'graph_only': (.5, False, False, False, False),
    'abundance_only': (0., True, False, True, False),
    'topology_only': (0., False, True, False, True),
    'graph_abundance': (.5, True, False, True, False),
    'graph_topology': (.5, False, True, False, True),
    'abundance_topology': (0., True, True, True, True),
    'no_abundance_products': (.5, True, True, False, True),
    'no_topology_products': (.5, True, True, True, False),
}


def variant_config(config, name):
    if name not in VARIANTS:
        raise ValueError(f'Unknown variant: {name}')
    result = dict(config)
    result.update(zip(['base_scale', 'i1', 'i2', 'product_i1', 'product_i2'], VARIANTS[name]))
    result['variant'] = name
    return result
