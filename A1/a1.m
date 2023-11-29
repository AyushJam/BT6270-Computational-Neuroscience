% Assignment 1 - BT6270
% EE20B018 Ayush Jamdar
% Aim - to calculate voltage oscillation frequency and impulse currents 
% at which the behaviour changes
% Hodgkin-Huxley Model 

% It is given that threshold current is 0.0223
nsamples = 1500;
I = linspace(0.001, 2.5, nsamples);
dt = 0.01; % from HHmodel
freqs = zeros(1, nsamples);
f_strength = zeros(1, nsamples);
num_aps = zeros(1, nsamples);

for i = 1:nsamples
    [v, ~, ~, ~] = HHmodel(I(i));
    [freqs(i), f_strength(i)] = get_frequency(v, dt); % method I
    num_aps(i) = get_num_peaks(v); % method II
end

% select only those signals with continuous firing
% this means that in an FFT, even aperiodic signals have weak fourier
% frequency components due to an inherent damped oscillatory behaviour
% these are not part of the limit cycle behaviour.

no_osc_freqs_indices = find(f_strength < 6.5); 
% this threshold was obtained
% from the power spectrum
freqs(no_osc_freqs_indices) = 0;
num_aps(no_osc_freqs_indices) = 0;

figure(1)
plot(I, freqs)
grid on
title("Voltage Frequency vs Impulse Current")
xlabel('Current in microamperes')
ylabel('Voltage Oscillation Frequency (Hz)')

figure(2)
plot(I, f_strength)
grid on
title('Power Spectrum')
xlabel('Current in microamperes')
ylabel('Fundamental frequency power')

figure(3)
plot(I, num_aps)
grid on
title('AP Frequency Plot')
xlabel('Current in microamperes')
ylabel('Number of APs')








