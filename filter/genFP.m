function genFP(file_name)
full_file_name = strcat('photos/', file_name);
%disp(file_name);
%disp(full_file_name);
orig = imread(full_file_name, 'jpg');
file_name(regexp(file_name,'[_]'))=[];
perm = randperm(length(file_name));
file_name =  file_name(perm);

% Resize image
scale = 700/min(size(orig, 1), size(orig, 2)); % scaling factor
normalized = imresize(orig, scale);
% normalized = normalized(((s(1)-sidelen)/2):((s(1)+sidelen-2)/2), ((s(2)-sidelen)/2):((s(2)+sidelen-2)/2), :);

% Increase contrast with RGB and HSV
rgb_im = zeros(size(normalized), 'uint8');
for ch = 1:3
    channel = normalized(:,:,ch);
    rgb_im(:,:,ch) = imadjust(channel, stretchlim(channel), []);
end
hsv_im = rgb2hsv(rgb_im);
% figure(1);
% imshow(hsv_im);
for ch = 2:3
    hsv_im(:,:,ch) = imadjust(hsv_im(:,:,ch), stretchlim(hsv_im(:,:,ch), 0.01), []);
end
% figure(2);
% imshow(hsv_im);

% Get rid of everything but the white and red parts of the picture
white_thresh_min = [0.0, 0.0, 0.4]; % HSV min threshold for white
white_thresh_max = [0.7, 0.6, 1.0]; % HSV max threshold for white
white_blobs = zeros(size(normalized, 1), size(normalized, 2));
for i = 1:size(normalized, 1)
    for j = 1:size(normalized, 2)
        temp = [hsv_im(i, j, 1), hsv_im(i, j, 2), hsv_im(i, j, 3)];
        if (temp >= white_thresh_min)
            if (temp <= white_thresh_max)
                white_blobs(i,j) = 1;
            end
        end
    end
end
% figure;
% imshow(white_blobs);

% Get rid of small blobs and specks for WHITE
%se = strel('disk', 3, 4);
%denoised = imopen(bw_blobs, se);
min_area = 100;
denoised_white = bwareaopen(white_blobs, min_area);
% figure;
% imshow(denoised_white);
% hold on

% Getting each individual WHITE blob
rp_white = regionprops(denoised_white, 'BoundingBox');
boxes_white = reshape([rp_white.BoundingBox], 4, []).';
grp = -1;
grp_idx = 0;
prev_grp_box = zeros(1,4);
delta = 10;
for idx = 1:numel(rp_white)
    box = boxes_white(idx, :); % [x y w h]
    % throw out blobs that clearly aren't numbers
    if (box(3) > box(4))
        continue;
    end
    box = round(box);
    wl = [box(1), box(2)];
    wh = [box(1) + box(3), box(2) + box(4)];
    % rectangle('Position', boxes_white(idx,:), 'EdgeColor', 'yellow');
    invert = ones(box(4), box(3));
    for i = box(2):(box(2)+box(4)-1)
        for j = box(1):(box(1)+box(3)-1)
            if (denoised_white(i, j) == 1)
                invert(i-box(2)+1,j-box(1)+1) = 0;
            end
        end
    end
    % Scale and center the number in a 200x200 square
    invert = imresize(invert, 170/size(invert, 1));
    export = ones(200);
    col = floor(100-size(invert,2)/2);
    export(15+(1:size(invert,1)),col:col+size(invert,2)-1) = invert;
    % Determine label as per schematic
    prev_on_left = abs(prev_grp_box(1)+prev_grp_box(3)-box(1));
    prev_on_right = abs(box(1)+box(3)-prev_grp_box(1));
    if (prev_grp_box(3) ~= 0 && (prev_on_left <= delta || prev_on_right <= delta))
        grp_idx = grp_idx + 1;
    else
        grp = grp + 1;
        grp_idx = 0;
    end
    %figure;
    %imshow(export);
    filename = strcat('output/', file_name, '_', int2str(grp), '_', int2str(grp_idx), '.jpg');
    imwrite(export, filename);
    prev_grp_box = box;
end
end