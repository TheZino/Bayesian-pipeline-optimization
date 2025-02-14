% Target:
%    Unpaired dataset Dt = {Lt}.
%    The final goal (not in this script) is to produce the high-light Ht version of low-light images Lt.
% Source:
%    Paired dataset Ds = {Ls, Hs}.
%    With this script, low-light images Ls will be mapped to Lm (the same low-light space of Dt).
% We compute the Source->Target mapping observing small images.
% We apply the learned mapping to large source images.

% SID-Sony to Night24
% source_small_path = 'data/SID/Sony/demosaiced_small_short/';
% source_large_path = 'data/SID/Sony/demosaiced_short/';
% target_small_path = 'data/Nightimaging/night24/demosaiced_small_short/';
% output_path = 'data/SID/Sony/Sony_remapped5/';

% SID-Fuji to Night24
source_small_path = 'data/SID/Fuji/demosaiced_small_short/';
source_large_path = 'data/SID/Fuji/demosaiced_short/';
target_small_path = 'data/Nightimaging/night24/demosaiced_small_short/';
output_path = 'data/SID/Fuji/Fuji_remapped5/';

mkdir(output_path);

%% Collect images

% Collect all small images from the source camera
source_small_files = dir(fullfile(source_small_path, '*.png'));
for f = 1:numel(source_small_files)
    fprintf('%d/%d (%.2f%%)\n', f, numel(source_small_files), 100*f/numel(source_small_files))
    img = imread(fullfile(source_small_path, source_small_files(f).name));
    if f==1
        source_small_imgs = img;
    else
        source_small_imgs = cat(2,source_small_imgs,img);
    end
end

% Collect all small images from the target camera
target_small_files = dir(fullfile(target_small_path, '*.png'));
for f = 1:numel(target_small_files)
    fprintf('%d/%d (%.2f%%)\n', f, numel(target_small_files), 100*f/numel(target_small_files))
    img = imread(fullfile(target_small_path, target_small_files(f).name));
    if f==1
        target_small_imgs = img;
    else
        img = imresize(img,[size(target_small_imgs, 1), NaN]);
        target_small_imgs = cat(2,target_small_imgs,img);
    end
end

%% Compute mapping transformation on small images

[mapped_small_imgs, ~, mappingFunction] = imhistmatch_train(source_small_imgs, target_small_imgs, 2^16, 'method', 'polynomial');

%% Apply mapping to large images from the source camera (load one at a time)
source_large_files = dir(fullfile(source_large_path, '*.png'));
for f = 1:numel(source_large_files)
    fprintf('%d/%d (%.2f%%)\n', f, numel(source_large_files), 100*f/numel(source_large_files))
    img = imread(fullfile(source_large_path, source_large_files(f).name));

    % Apply learned mapping to one large source image
    mapped_img = imhistmatch_test(img, mappingFunction, 2^16, 'method', 'polynomial');
    
    imwrite(mapped_img, fullfile(output_path, source_large_files(f).name));
end

%% Apply mapping to large images from the source camera (load all at once - do not use)

% % Collect all large images from the source camera
% source_large_files = dir(fullfile(source_large_path, '*.png'));
% for f = 1:numel(source_large_files)
%     fprintf('%d/%d (%.2f%%)\n', f, numel(source_large_files), 100*f/numel(source_large_files))
%     img = imread(fullfile(source_large_path, source_large_files(f).name));
%     if f==1
%         source_large_imgs = img;
%     else
%         source_large_imgs = cat(2,source_large_imgs,img);
%     end
% end

% % Apply learned mapping to all large source images
% mapped_large_imgs = imhistmatch_test(source_large_imgs, mappingFunction, 2^16, 'method', 'polynomial');

% % Save all mapped large images
% left = 1;
% for f = 1:numel(source_large_files)
%     fprintf('%d/%d (%.2f%%)\n', f, numel(source_large_files), 100*f/numel(source_large_files))
%     img = imread(fullfile(source_large_path, source_large_files(f).name));
    
%     width = size(img,2);
%     right = left+width-1;
%     mapped_img = mapped_large_imgs(:,left:right,:);
%     left = left+width;
    
%     imwrite(mapped_img, fullfile(output_path, source_large_files(f).name));
% end
