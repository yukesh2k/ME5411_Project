DISPLAY_TASK = 6;

disp('Task 1: ');
image = imread('charact2.bmp');

% RGB channel specific contrast adjustment
red_channel = image(:, :, 1);
green_channel = image(:, :, 2);
blue_channel = image(:, :, 3);

if (DISPLAY_TASK == 1)
    red_channel_imadjust = imadjust(red_channel);
    red_channel_histeq = histeq(red_channel);
    red_channel_adapthisteq = adapthisteq(red_channel);
    
    green_channel_imadjust = imadjust(green_channel);
    green_channel_histeq = histeq(green_channel);
    green_channel_adapthisteq = adapthisteq(green_channel);
    
    blue_channel_imadjust = imadjust(blue_channel);
    blue_channel_histeq = histeq(blue_channel);
    blue_channel_adapthisteq = adapthisteq(blue_channel);
    
    image_imadjust = cat(3, red_channel_imadjust, green_channel_imadjust, blue_channel_imadjust);
    image_histeq = cat(3, red_channel_histeq, green_channel_histeq, blue_channel_histeq);
    image_adapthisteq = cat(3, red_channel_adapthisteq, green_channel_adapthisteq, blue_channel_adapthisteq);

    montage({image,image_imadjust,image_histeq,image_adapthisteq},"Size",[1 4])
    title("Original Image and Contrast Enhanced Images using imadjust, histeq, and adapthisteq")
end

disp('Task 2')

function average_filtered_image = applyAveragingFilter(red_channel, green_channel, blue_channel, n1, n2)
    kernel = ones(n1, n2) / (n1 * n2);
    
    red_filtered = imfilter(red_channel, kernel, 'same');
    green_filtered = imfilter(green_channel, kernel, 'same');
    blue_filtered = imfilter(blue_channel, kernel, 'same');
    
    average_filtered_image = cat(3, red_filtered, green_filtered, blue_filtered);
end

if (DISPLAY_TASK == 2)
    average_filtered_image_3_3 = applyAveragingFilter(red_channel, green_channel, blue_channel, 3, 3)
    average_filtered_image_5_5 = applyAveragingFilter(red_channel, green_channel, blue_channel, 5, 5)
    average_filtered_image_7_7 = applyAveragingFilter(red_channel, green_channel, blue_channel, 7, 7)
    average_filtered_image_9_9 = applyAveragingFilter(red_channel, green_channel, blue_channel, 9, 9)
    average_filtered_image_16_16 = applyAveragingFilter(red_channel, green_channel, blue_channel, 16, 16)

    montage({average_filtered_image_3_3,average_filtered_image_5_5,average_filtered_image_7_7,average_filtered_image_9_9, average_filtered_image_16_16},"Size",[1 5])
    title("3 x 3, 5 x 5, 7 x 7, 9 x 9, 16 x 16 kernel Average Filtered images")
end

disp('Task 3: ');

image_gray = image(:, :, 2);

[M, N] = size(image_gray); 
    
FT_img = fft2(double(image_gray)); 
  
threshold_freq = 0.5; 
  
u = 0:(M-1); 
idx = find(u>M/2); 
u(idx) = u(idx)-M; 
v = 0:(N-1); 
idy = find(v>N/2); 
v(idy) = v(idy)-N; 
  
[V, U] = meshgrid(v, u); 
  
D = sqrt(U.^2+V.^2); 
  
H = double(D > threshold_freq); 
  
G = H.*FT_img; 

image_high_pass_filtered = real(ifft2(double(G)));

if (DISPLAY_TASK == 3)

    figure;
    subplot(1, 2, 1), imshow(image_gray), title('Original');
    subplot(1, 2, 2), imshow(image_high_pass_filtered, [ ]), title('High Pass Filter');

end

disp('Task 4');
[height, width] = size(image_high_pass_filtered);
midpoint = round(height)/2;

sub_image = image_high_pass_filtered(midpoint+1:end, :);

if (DISPLAY_TASK == 4)
    figure;
    subplot(1, 1, 1), imshow(sub_image)
end

disp('Task 5');
binary_image = imbinarize(sub_image);

if (DISPLAY_TASK == 5)
    figure;
    subplot(1, 1, 1), imshow(binary_image);
end

disp('Task 6');
outline_image = edge(binary_image, 'canny');

if (DISPLAY_TASK == 6)
    figure;
    subplot(1, 1, 1), imshow(outline_image);
end
