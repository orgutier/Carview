clear all; close all; clc;
%% Capturamos las imágenes que estarán en nuestra base de datos

% Creamos un objeto que detecta los rostros y establecemos el mínimo tamaño
% detectable de un rostro en 150x150 pixeles
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART','MinSize',[150,150]);


% Crea un objeto que servirá para rastrear los puntos en la imagen
% utilizando el método de umbral de error de ida y vuelta en diferentes
% imágenes. Con umbral de error de 2 pixeles
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Create un objeto para la webcam
%cam = webcam();
cam = ipcam('http://192.168.1.148:8080/video');
% Capture una imagen para calcular el tamaño
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

% Crea un objeto para mostrar la imagen en un video y se establecen el
% tamaño y la posición de la ventana de video [100 100 670 510]
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
runLoop = true; % Variable para corroborar si la ventana del video sigue abierta
numPts = 0; % Variable para contar el número de puntos detectados en un objeto
i=1; 


% Mientras la ventana de el video player esté abierta y no se alcance el
% número de fotos establecidas, que siga contando
while runLoop 
    % Get the next frame.
    videoFrame = snapshot(cam); % Toma las fotos
    videoFrameGray = rgb2gray(videoFrame); % Convierte a escala de grises
    
    % Si el número de puntos es menor que 10 busca un rostro en la imagen
    if numPts < 10
        % Guarda la ubicación y las medidas de la cara detectada en la imagen
        % [x y width height]
        bbox = faceDetector.step(videoFrameGray);
        
        % Si no está vacío el vector de la ubicación de a cara, significa
        % que sí encontró un rostro
        if ~isempty(bbox)

           
            % Crea un objeto con las  características de los puntos 
            % detectados en la ROI de la imagen
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bbox(1, :));
            
            % Guarda las posiciones de los puntos detectados
            xyPoints = points.Location;
            % Número de puntos detectados
            numPts = size(xyPoints,1);
            % Re-inicializa el objeto point tracker
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            % Guarda los puntos anteriores
            oldPoints = xyPoints;
            
            % Convierte el rectangulo con las posiciones [x, y, w, h] en
            % una matriz M-por-2 de las coordenadas [x,y] de las 4 esquinas
            % Se necesita esto para poder transformar la caja de posiciones
            %  y mostrar la orientación del rostro en una imagen
            bboxPoints = bbox2points(bbox(1, :));
            
            % Convierte la caja con las esquinas en un vector con el formato
            % [x1 y1 x2 y2 x3 y3 x4 y4] requerido para la función insertShape
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Muestra un cuadrado al rededor del rostro detectado
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Muestra los puntos característicos detectados con unas cruces
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'green');
        end
        
    else % Si encuentra más de 9 puntos característicos en la imagen 
        
        % Busca puntos y guarda la ubicación de cada uno y se anota si
        % existe en la imagen
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        
        % Guarda los puntos detectados visibles en la imagen
        visiblePoints = xyPoints(isFound, :);
        
        % Guarda los puntos anteriores que coinciden con los puntos
        % detectados en esta nueva búsqueda
        oldInliers = oldPoints(isFound, :);
        
        % Calcula el número de puntos detectados
        numPts = size(visiblePoints, 1);
        
        if numPts >= 10
            % Estima la transformación geométrica entre los puntos viejos y
            % los puntos nuevos (Revisa la similaridad entre los dos, calculando
            % su distancia máxima = 4 pixeles)
            [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
                oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            
            % Aplica la transformación a la caja de coordenadas (la
            % actualiza con los nuevos valores de coordenadas)
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            % Convierte la caja de coordenadas en un vector con el formato
            % [x1 y1 x2 y2 x3 y3 x4 y4] requerido para insertShape
            bboxPolygon = reshape(bboxPoints', 1, []);
            
            % Muestra un cuadrado al rededor del rostro encontrado
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            
            % Muestra los puntos encontrados en la imagen con cruces blancas
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'green');
            
            % Resetea los puntos y los guarda en la variable de puntos viejos
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
            i = i+1;
            
        end
    end
    
    % Muestra el video con las anotaciones usando el video player object
    step(videoPlayer, videoFrame);
    
    % Revisa si la ventana de video player se ha cerrado
    runLoop = isOpen(videoPlayer);
end


% Limpia las variables
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);

