function varargout = imhistmatch_test(varargin)
%IMHISTMATCH Adjust 2-D image to match its histogram to that of another image.
%
%   B = IMHISTMATCH(A,REF) transforms the input grayscale or truecolor
%   image A so that the histogram of the output image B approximately
%   matches the histogram of the reference image REF. For truecolor images,
%   each color channel of A is matched independently to the corresponding
%   color channel of REF.
%
%   B = IMHISTMATCH(A, REF, NBINS) uses NBINS equally spaced histogram bins
%   for transforming input image A. The image returned in B has no more
%   than NBINS discrete levels. The default value for NBINS is 64.
%
%   [B, HGRAM] = IMHISTMATCH(A, REF,...) also returns the histogram of the
%   reference image REF used for matching in HGRAM. HGRAM is a 1 x NBINS
%   (when REF is grayscale) or a 3 x NBINS (when REF is truecolor) matrix,
%   where NBINS is the number of histogram bins. Each row in HGRAM stores
%   the histogram of a single color channel of REF.
%
%   [___] = IMHISTMATCH(___, PARAM, VAL) changes the behavior of the
%   histogram matching algorithm using the following name-value pairs.
%
%       'Method'    The technique used to map REF image to image A:
%                   'uniform' (default) or 'polynomial'. The 'uniform'
%                   method uses a histogram-based intensity function and
%                   histogram equalization. The 'polynomial' method employs
%                   a smooth intensity mapping function and cubic hermite
%                   polynomial fitting.
%
%   Notes
%   -----
%   1. The histograms for A and REF are computed with equally spaced bins
%      and with intensity values in the appropriate range for each image:
%      [0,1] for images of class double or single, [0,255] for images of
%      class uint8, [0,65535] for images of class uint16, and [-32768,
%      32767] for images of class int16.
%
%   Class Support
%   -------------
%   A can be uint8, uint16, int16, double or single. The output image B has
%   the same class as A. The optional output HGRAM is always of class double.
%
%   Example 1
%   -------
%   % This example matches the histogram of one image to that of another image
%   % using uniform method.
%
%   A = imread('office_2.jpg');
%   figure, imshow(A, []);
%   title('Original Image');
%
%   ref = imread('office_4.jpg');
%   figure, imshow(ref, []);
%   title('Reference Image');
%
%   B = imhistmatch(A, ref);
%
%   figure, imshow(B, []);
%   title('Histogram Matched Image Using Uniform Method');
%
%   Example 2
%   ---------
%   % This example matches the histogram of one image to that of another image
%   % using polynomial method
%
%   A = imread('office_2.jpg');
%   figure, imshow(A, []);
%   title('Original Image');
%
%   ref = imread('office_4.jpg');
%   figure, imshow(ref, []);
%   title('Reference Image');
%
%   B = imhistmatch(A, ref, 'method', 'polynomial');
%
%   figure, imshow(B, []);
%   title('Histogram Matched Image Using Polynomial Method');
%
%   Example 3
%   ---------
%   % This example compares the histogram of one image to that of another image
%   % using polynomial and uniform methods.
%
%   A = imread('office_4.jpg');
%   figure, imshow(A, []);
%   title('Original Image');
%
%   ref = imread('office_2.jpg');
%   figure, imshow(ref, []);
%   title('Reference Image');
%
%   BPoly = imhistmatch(A, ref, 'method', 'polynomial');
%   BUni = imhistmatch(A, ref, 'method', 'uniform');
%   figure, montage({BPoly, BUni});
%   title('Histogram Matched Image Comparison Between Polynomial and Uniform Methods');
%
%   References:
%   ---------
%   [1] M.D.Grossberg, S.K.Nayar, Determining the camera response from
%   images: What is knowable?, PAMI, 2003.
%
%   See also HISTEQ, IMADJUST, IMHIST, IMHISTMATCHN.

%   Copyright 2012-2018 The MathWorks, Inc.

narginchk(2,5);
nargoutchk(0,2);

[A, tranFunction, N, method] = parse_inputs(varargin{:});
% If input A is empty, then the output image B will also be empty

originalClass = class(A);
numColorChan = size(A, 3);

switch method
    
    case 'uniform'
        error('Uniform not yet implemented');
        isColor = numColorChan > 1;
        
        % Compute histogram of the reference image
        hgram = zeros(numColorChan, N);
        for currentChan = 1:numColorChan
            hgram(currentChan, :) = imhist(ref(:,:,currentChan), N);
        end
        
        % Adjust A using reference histogram
        hgramToUse = 1;
        for k = 1:size(A, 3) % Process one color channel at a time
            if isColor
                hgramToUse = k; % Use the k-th color channel's histogram
            end
            
            for p = 1:size(A, 4)
                % Use A to store output, to save memory
                A(:, :, k, p) = histeq(A(:,:,k,p), hgram(hgramToUse,:));
            end
        end
        
        % Always set varargout{1} so 'ans' always gets
        % populated even if user doesn't ask for output
        varargout{1}  = A;
        if (nargout == 2)
            varargout{2} = hgram;
        end
        
    case 'polynomial'
        
        if isfloat(A)
            
            % The algorithm assumes the input image is in [0,1]
            A = min(1, max(0, A));
        else
            
            % convert to single if image is not float type
            A = im2single(A);
        end
        
        % Transform color
        intensityMappedOutput = zeros(size(A));
        for currentChan = 1:length(tranFunction)
            intensityMappedOutput(:,:,currentChan) = ppval(tranFunction{currentChan}, A(:,:,currentChan));
        end
        varargout{1} = convertToOriginalClass(intensityMappedOutput, originalClass);
        
        if (nargout >= 2)
            varargout{2} = [];
        end

end

end

function B = convertToOriginalClass(B, OriginalClass)

if strcmp(OriginalClass,'uint8')
    B = im2uint8(B);
elseif strcmp(OriginalClass,'uint16')
    B = im2uint16(B);
elseif strcmp(OriginalClass,'single')
    B = im2single(B);
    B = min(1, max(0, B));
elseif strcmp(OriginalClass,'int16')
    %  double
    B = im2int16(B);
else
    B = (min(1, max(0, B)));
end
end

%--------------------------------------------------------------------------
function [A, tranFunction, N,method] = parse_inputs(varargin)

parser = inputParser;
parser.FunctionName = mfilename;
parser.CaseSensitive = false;
parser.PartialMatching = true;

A = varargin{1};
validateattributes(A,{'uint8','uint16','double','int16', ...
    'single'},{'nonsparse','real'}, mfilename,'A',1);

tranFunction = varargin{2};

parser.addOptional('bins', 64, @checkBins);
parser.addParameter('method','uniform',@checkMethod);

parser.parse(varargin{3:end});

N = parser.Results.bins;
method = validatestring(parser.Results.method, {'uniform', 'polynomial'}, ...
    mfilename, 'method');

if strcmp(method, 'polynomial')
    
    if isscalar(A)
        error(message('images:imhistmatch:expectedArray'));
    end
    
    if (ndims(A) > 3 || size(A,3) > 3)
        error(message('images:imhistmatch:invalidNumberOfImagesChannels'));
    end
end
end

function tr = checkBins(bins)
validateattributes(bins, ...
    {'numeric'}, ...
    {'scalar','nonsparse','integer','>', 1}, ...
    mfilename,'bins',3);
tr = true;
end

function tf = checkMethod(methodString)
validateattributes(methodString, ...
    {'char', 'string'},...
    {'scalartext'},...
    mfilename, 'method');
tf = true;
end
