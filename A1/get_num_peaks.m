% EE20B018 A1

function n = get_num_peaks(v)
    peak_indices = find(islocalmax(v) == 1);
    n = length(peak_indices);
    % note that this also counts the little maxima that are not APs
    % they are usually lesser than the number of APs but occur in all
    % so they don't affect relative rates
end