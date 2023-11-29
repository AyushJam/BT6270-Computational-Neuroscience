% EE20B018 A1

function [freq, strength] = get_frequency(v, dt)
    fft_v = fft(v)/10000;
    A = abs(fft_v).^2; % power spectrum

    % % this part doesn't always work due to tiny local maxima
    % [~, index_max_power] = max(A(2:end)); % exclude the DC
    % index_max_power = index_max_power + 1; 
    % % because a sliced array was used 

    local_maxes = find(islocalmax(A) == 1);
    
    % this part selects the first AP point
    % by ignoring the little maxima that occurs before the AP
    if abs(A(local_maxes(1)) - A(local_maxes(1)+1)) >= 1 
        index_max_power = local_maxes(1); 
    else 
        index_max_power = local_maxes(2);
    end

    % % debugging tools
    % display(A(index_max_power))
    % display(index_max_power)
    % plot(A)
    % grid on
    % title('DFT of Voltage')
    strength = A(index_max_power);
    freq = index_max_power / (dt * length(A));
end