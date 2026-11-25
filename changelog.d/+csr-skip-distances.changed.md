`lc_pairs_within_csr` computes no distances when `out_d2` is null. Rows kept from a short buffer with a null `out_d2` hold no distances, so a next call that asks for them searches again.
