import numpy as np
import cv2

class cvplt:
    """Plots NumPy arrays onto BGR image arrays for display with cv2.imshow()."""

    # --- Public API ---

    def draw_plot(data, renderArray=None, plotBeginXY=None, plotEndXY=None,
                  plotTitle="", plotBackgroundColour=[2,2,2],
                  plotOutlineColour=[250,250,250], plotValuesColour=[250,250,250]):
        """
        Overlay a 1-D data series onto a BGR render array.

        Adjacent non-NaN samples are drawn as lines; isolated non-NaN samples
        as single-pixel circles.  NaN values are skipped.

        Args:
            data (np.ndarray): 1-D array of values.  May contain np.nan.
            renderArray (np.ndarray, optional): Destination BGR or grayscale
                image.  Created automatically (min 640x480) if None.
            plotBeginXY (list, optional): Top-left [x, y] of the plot region.
            plotEndXY (list, optional): Bottom-right [x, y] of the plot region.
            plotTitle (str): Label shown next to the max value.
            plotBackgroundColour (list|int): BGR fill colour. Default [2,2,2].
            plotOutlineColour (list|int): BGR border/text colour. Default [250,250,250].
            plotValuesColour (list|int): BGR data colour. Default [250,250,250].

        Returns:
            np.ndarray: renderArray with the plot composited onto it.
        """
        dataLen = len(data)
        dataCount = np.count_nonzero(~np.isnan(data))
        # Guard: if data is empty or all-NaN, return early before any nanmin/nanmax calls.
        if dataLen < 1 or dataCount < 1:
            if renderArray is None:
                renderArray = cvplt._get_screenarray_colour([640, 480], plotBackgroundColour)
            return renderArray
        if (renderArray is None):
            dataMin = np.nanmin(data)
            dataMax = np.nanmax(data)
            dataLenForRenderArray = np.nanmax([640, dataLen])
            dataRange = np.nanmax([480, np.ceil(abs(dataMax - dataMin))])
            renderArraySize = np.array([dataLenForRenderArray, dataRange], dtype="int")
            renderArray = cvplt._get_screenarray_colour(renderArraySize, plotBackgroundColour)
        # Determine Grayscale or Colour
        renderArrayShapeLen = len(renderArray.shape)
        if (renderArrayShapeLen == 2):
            grayOrColour = False
        elif (renderArrayShapeLen == 3):
            grayOrColour = True
        else:
            return renderArray
        # Convert Colours if necessary
        if not grayOrColour:
            if not isinstance(plotBackgroundColour, (np.ndarray, int)):
                plotBackgroundColour = int(np.mean(plotBackgroundColour))
            if not isinstance(plotOutlineColour, (np.ndarray, int)):
                plotOutlineColour = int(np.mean(plotOutlineColour))
            if not isinstance(plotValuesColour, (np.ndarray, int)):
                plotValuesColour = int(np.mean(plotValuesColour))
        else:
            if isinstance(plotBackgroundColour, (np.ndarray, int)):
                plotBackgroundColour = np.array([plotBackgroundColour] * 3, dtype="int")
            if isinstance(plotOutlineColour, (np.ndarray, int)):
                plotOutlineColour = np.array([plotOutlineColour] * 3, dtype="int")
            if isinstance(plotValuesColour, (np.ndarray, int)):
                plotValuesColour = np.array([plotValuesColour] * 3, dtype="int")
        # Get plotArrayPosition and plotArraySize
        if plotBeginXY is None or plotEndXY is None:
            plotBeginXY = np.zeros(2, dtype="int")
            plotEndXY = cvplt._get_screensize(renderArray)
        plotBeginXY = np.array(plotBeginXY)
        plotEndXY = np.array(plotEndXY)
        plotArraySize = np.array(plotEndXY-plotBeginXY, dtype="int")
        if plotArraySize[0] <= 0 or plotArraySize[1] <= 0:
            return renderArray
        data, dataLen = cvplt._data_resize(data, dataLen, plotArraySize)
        plotArraySizeHalf = np.array(plotArraySize * 0.5, dtype="int")
        plotArrayPosition = plotBeginXY + plotArraySizeHalf
        plotArray, plotArraySize = cvplt._plotArray_create(plotArraySize, grayOrColour, plotBackgroundColour)
        # Inner area: 1 px inset on each side to stay within the outline border.
        plotArraySizeActual = np.array([plotArraySize[0]-2, plotArraySize[1]-2])
        dataToPlot, dataLenToPlot, dataRange = cvplt._plotArray_process_data(data, dataLen, plotArraySizeActual)
        plotArray = cvplt._plotArray_draw_plot(plotArray, plotArraySizeActual, dataToPlot, dataLenToPlot, plotValuesColour)
        plotArray = cvplt._plotArray_draw_text(plotTitle, plotArray, plotArraySize, dataRange, plotOutlineColour)
        plotArray = cvplt._plotArray_draw_outline(plotArray, plotArraySize, plotOutlineColour)
        renderArray = cvplt._draw_plotArray_to_renderArray(plotArray, renderArray, plotArrayPosition, plotArraySize)
        return renderArray
        
    def draw_plot_coords(data, renderArray=None, connectDots=False, plotBeginXY=None, plotEndXY=None,
                         plotTitle="", plotBackgroundColour=[2,2,2],
                         plotOutlineColour=[250,250,250], plotValuesColour=[250,250,250]):
        """
        Overlay 2-D coordinate data onto a BGR render array.

        Each coordinate is drawn as a filled circle.  All coordinate values
        must be finite integers (no NaNs).

        Args:
            data (np.ndarray): N×2 integer array of [x, y] coordinates.
            renderArray (np.ndarray, optional): Destination image.  Created
                automatically (min 640x480) if None.
            connectDots (bool): Connect consecutive points. *Not yet implemented.*
            plotBeginXY (list, optional): Top-left [x, y] of the plot region.
            plotEndXY (list, optional): Bottom-right [x, y] of the plot region.
            plotTitle (str): Label shown next to the coordinate range.
            plotBackgroundColour (list|int): BGR fill colour. Default [2,2,2].
            plotOutlineColour (list|int): BGR border/text colour. Default [250,250,250].
            plotValuesColour (list|int): BGR dot colour. Default [250,250,250].

        Returns:
            np.ndarray: renderArray with the plot composited onto it.
        """
        dataLen = len(data)
        # Guard: nothing to plot if data is empty.
        if dataLen < 1:
            if renderArray is None:
                renderArray = cvplt._get_screenarray_colour([640, 480], plotBackgroundColour)
            return renderArray
        if (renderArray is None):
            dataMinX = min(data, key=lambda x: x[0])[0]
            dataMinY = min(data, key=lambda x: x[1])[1]
            dataMaxX = max(data, key=lambda x: x[0])[0]
            dataMaxY = max(data, key=lambda x: x[1])[1]
            dataRangeX = np.nanmax([640, np.ceil(abs(dataMaxX - dataMinX))])
            dataRangeY = np.nanmax([480, np.ceil(abs(dataMaxY - dataMinY))])
            renderArraySize = np.array([dataRangeX, dataRangeY], dtype="int")
            renderArray = cvplt._get_screenarray_colour(renderArraySize, plotBackgroundColour)
        # Detect grayscale (2-D) vs colour (3-D) array.
        renderArrayShapeLen = len(renderArray.shape)
        if (renderArrayShapeLen == 2):
            grayOrColour = False
        elif (renderArrayShapeLen == 3):
            grayOrColour = True
        else:
            return renderArray
        # Normalise colour args to match the array's colour space.
        if not grayOrColour:
            if not isinstance(plotBackgroundColour, (np.ndarray, int)):
                plotBackgroundColour = int(np.mean(plotBackgroundColour))
            if not isinstance(plotOutlineColour, (np.ndarray, int)):
                plotOutlineColour = int(np.mean(plotOutlineColour))
            if not isinstance(plotValuesColour, (np.ndarray, int)):
                plotValuesColour = int(np.mean(plotValuesColour))
        else:
            if isinstance(plotBackgroundColour, (np.ndarray, int)):
                plotBackgroundColour = np.array([plotBackgroundColour] * 3, dtype="int")
            if isinstance(plotOutlineColour, (np.ndarray, int)):
                plotOutlineColour = np.array([plotOutlineColour] * 3, dtype="int")
            if isinstance(plotValuesColour, (np.ndarray, int)):
                plotValuesColour = np.array([plotValuesColour] * 3, dtype="int")
        if plotBeginXY is None or plotEndXY is None:
            plotBeginXY = np.zeros(2, dtype="int")
            plotEndXY = cvplt._get_screensize(renderArray)
        plotBeginXY = np.array(plotBeginXY)
        plotEndXY = np.array(plotEndXY)
        plotArraySize = np.array(plotEndXY-plotBeginXY, dtype="int")
        if plotArraySize[0] <= 0 or plotArraySize[1] <= 0:
            return renderArray
        plotArraySizeHalf = np.array(plotArraySize * 0.5, dtype="int")
        plotArrayPosition = plotBeginXY + plotArraySizeHalf
        plotArray, plotArraySize = cvplt._plotArray_create(plotArraySize, grayOrColour, plotBackgroundColour)
        # Inner area: 1 px inset on each side to stay within the outline border.
        plotArraySizeActual = np.array([plotArraySize[0]-2, plotArraySize[1]-2])
        data, dataRange, _, dataBufferSize, _ = cvplt._data_resize_calibrate_coords(data, plotArraySizeActual)
        plotArray = cvplt._plotArray_draw_plot_coords(plotArray, plotArraySizeActual, data, dataBufferSize, connectDots, plotValuesColour)
        plotArray = cvplt._plotArray_draw_text_coords(plotTitle, plotArray, plotArraySize, dataRange, plotOutlineColour)
        plotArray = cvplt._plotArray_draw_outline(plotArray, plotArraySize, plotOutlineColour)
        renderArray = cvplt._draw_plotArray_to_renderArray(plotArray, renderArray, plotArrayPosition, plotArraySize)
        return renderArray

    # --- Private helpers ---
    def _get_screensize(screenshot):
        screenshape = screenshot.shape
        screensize = np.array([screenshape[1], screenshape[0]], dtype="int")
        return screensize

    def _get_screencenter(screensize):
        screencenter = (int(screensize[0]*0.5), int(screensize[1]*0.5))
        return screencenter
    
    def _get_screenarray_gray(screensize, backgroundColour):
        if not isinstance(backgroundColour, (np.ndarray, int)):
            backgroundColour = np.mean(backgroundColour)
        screenArray = np.zeros((screensize[1], screensize[0]), dtype = 'uint8')
        screenArray[:][:] = backgroundColour
        return screenArray

    def _get_screenarray_colour(screensize, backgroundColour):
        if isinstance(backgroundColour, (np.ndarray, int)):
            backgroundColour = np.array([backgroundColour] * 3, dtype="int")
        screenArray = np.zeros((screensize[1], screensize[0], 3), dtype = 'uint8')
        screenArray[:][:] = backgroundColour
        return screenArray
    
    # Plot-Related Functions
    def _data_resize(data, dataLen, plotArraySize):
        if (dataLen == plotArraySize[0]):
            return data, dataLen
        data = data.astype("float32")
        data = cv2.resize(data, [1,plotArraySize[0]])
        dataLen = len(data)
        data = data.reshape(-1)
        return data, dataLen
    
    def _data_resize_calibrate_coords(data, plotArraySize):
        # Original bounds
        dataMinX = min(data, key=lambda x: x[0])[0]
        dataMinY = min(data, key=lambda x: x[1])[1]
        dataMaxX = max(data, key=lambda x: x[0])[0]
        dataMaxY = max(data, key=lambda x: x[1])[1]
        dataRange = np.array([[dataMinX, dataMinY],[dataMaxX, dataMaxY]], dtype="int")
        dataSizeX = dataMaxX - dataMinX
        dataSizeY = dataMaxY - dataMinY
        dataSize = np.array([dataSizeX, dataSizeY], dtype = "int")
        dataBufferSingle = np.max([int(dataSizeX * 0.1), int(dataSizeY * 0.1)]) # 10% margin on each side
        dataBufferSize = np.array([dataBufferSingle, dataBufferSingle], dtype="int")
        dataTotalSizeX = max(dataSizeX + (dataBufferSingle * 2), 1)  # guard: avoid div/0 when all coords are identical
        dataTotalSizeY = max(dataSizeY + (dataBufferSingle * 2), 1)
        dataTotalSize = np.array([dataTotalSizeX, dataTotalSizeY], dtype="int")
        dataMultiplierX = plotArraySize[0] / dataTotalSize[0]
        dataMultiplierY = plotArraySize[1] / dataTotalSize[1]
        dataMultipliers = np.array([dataMultiplierX, dataMultiplierY], dtype="float")
        # Shift origin to (0,0) and scale to plot-pixel space.
        dataCorrection = np.array([-dataMinX, -dataMinY], dtype="int")
        for c, coord in enumerate(data):
            data[c] = (coord + dataCorrection) * dataMultipliers
        # Recalculate bounds after rescaling.
        newDataBufferSize = (dataBufferSize * dataMultipliers).astype(int)
        dataMinX = min(data, key=lambda x: x[0])[0]
        dataMinY = min(data, key=lambda x: x[1])[1]
        dataMaxX = max(data, key=lambda x: x[0])[0]
        dataMaxY = max(data, key=lambda x: x[1])[1]
        dataSizeX = dataMaxX - dataMinX
        dataSizeY = dataMaxY - dataMinY
        newDataSize = np.array([dataSizeX, dataSizeY], dtype = "int")
        newDataTotalSizeX = dataSizeX + (newDataBufferSize[0] * 2)
        newDataTotalSizeY = dataSizeY + (newDataBufferSize[1] * 2)
        newDataTotalSize = np.array([newDataTotalSizeX, newDataTotalSizeY], dtype="int")
        return data, dataRange, newDataSize, newDataBufferSize, newDataTotalSize
        
    def _plotArray_create(plotArraySize, grayOrColour, plotBackgroundColour):
        if not grayOrColour:
            plotArray = cvplt._get_screenarray_gray(plotArraySize, plotBackgroundColour)
        else:
            plotArray = cvplt._get_screenarray_colour(plotArraySize, plotBackgroundColour)
        plotArraySize = cvplt._get_screensize(plotArray)
        return plotArray, plotArraySize
    
    def _plotArray_draw_outline(plotArray, plotArraySize, plotOutlineColour):
        # Draw plotOutlines on plotArray
        plotArray = cv2.rectangle(plotArray, 
                np.array([0, 0]),
                np.array([plotArraySize[0]-1, plotArraySize[1]-1]),
                plotOutlineColour, 1)
        return plotArray
    
    def _plotArray_process_data(data, dataLen, plotArraySizeActual):
        dataNanCount = np.count_nonzero(np.isnan(data))
        dataSortedMagnitude = data.copy()
        dataSortedMagnitude.sort()
        dataLenToPlot = dataLen
        if dataLen > plotArraySizeActual[0]:
            dataLenToPlot = plotArraySizeActual[0]
        dataRange = np.array([dataSortedMagnitude[0], dataSortedMagnitude[-1-dataNanCount]])
        dataRangeLen = dataRange[1]-dataRange[0]
        # Multiply Data Values (Y) to plotArray Y size
        if (dataRangeLen < 1):
            dataMultiplier = 1
        else:
            if (plotArraySizeActual[1] % 2 == 0): # even height: shrink by 1 px to stay inside outline
                dataMultiplier = (plotArraySizeActual[1]-1) / dataRangeLen
            else:
                dataMultiplier = plotArraySizeActual[1] / dataRangeLen
        dataMultiplied = (data * dataMultiplier)
        dataMultipliedSortedMagnitude = dataMultiplied.copy()
        dataMultipliedSortedMagnitude.sort()
        dataMultipliedRange = [int(np.floor(dataMultipliedSortedMagnitude[0])), int(np.ceil(dataMultipliedSortedMagnitude[-1-dataNanCount]))]
        dataToPlot = dataMultiplied
        # Shift values so the minimum maps to y=0.
        if (dataMultipliedRange[0] < 0):
            dataToPlot = dataToPlot + np.absolute(dataMultipliedRange[0])
        elif (dataMultipliedRange[0] > 0):
            dataToPlot = dataToPlot - dataMultipliedRange[0] - 1
        if (plotArraySizeActual[1] % 2 == 0):
            dataToPlot = dataToPlot + 1
        return dataToPlot, dataLenToPlot, dataRange
    
    def _plotArray_draw_plot(plotArray, plotArraySizeActual, dataToPlot, dataLenToPlot, plotValuesColour):
        # Iterate right-to-left; draw a line between adjacent non-NaN samples,
        # or a dot when one neighbour is NaN.
        if (dataLenToPlot > 1):
            for i in range(1,dataLenToPlot):
                values = np.array([dataToPlot[-i-1], dataToPlot[-i]])
                coordinates = np.zeros((2,2), dtype="int")
                nanCount = np.count_nonzero(np.isnan(values))
                if (nanCount == 0): # both samples valid — draw line
                    coordinates[0][0] = plotArraySizeActual[0]-i+1 # posterior
                    coordinates[0][1] = plotArraySizeActual[1]-int(values[1])
                    coordinates[1][0] = plotArraySizeActual[0]-i   # anterior
                    coordinates[1][1] = plotArraySizeActual[1]-int(values[0])
                    plotArray = cv2.line(plotArray, coordinates[0], coordinates[1], plotValuesColour, 1)
                elif (nanCount == 1):
                    if np.isnan(values[0]): # anterior is NaN — dot at posterior
                        coordinates[1][0] = plotArraySizeActual[0]-i+1
                        coordinates[1][1] = plotArraySizeActual[1]-int(values[1])
                        plotArray = cv2.circle(plotArray, coordinates[1], radius=0, color=plotValuesColour, thickness=1)
                    else:                   # posterior is NaN — dot at anterior
                        coordinates[0][0] = plotArraySizeActual[0]-i
                        coordinates[0][1] = plotArraySizeActual[1]-int(values[0])
                        plotArray = cv2.circle(plotArray, coordinates[0], radius=0, color=plotValuesColour, thickness=1)
                else:
                    pass
        else:
            val = float(dataToPlot[0])
            if not np.isnan(val):
                dotSize = int(max(plotArraySizeActual[0] * 0.2, plotArraySizeActual[1] * 0.2, 1))
                coordinates = np.array([0, int(val)], dtype="int")
                plotArray = cv2.circle(plotArray, coordinates, radius=dotSize, color=plotValuesColour, thickness=-1)
        return plotArray
    
    def _plotArray_draw_plot_coords(plotArray, plotArraySizeActual, data, dataBufferSize, connectDots, plotValuesColour):
        dotSize = int(np.nanmax([plotArraySizeActual[0] * 0.015, plotArraySizeActual[1] * 0.015, 1]))
        coordinates = np.zeros(2, dtype="int")
        for i, coord in enumerate(data):
            coordinates[0] = coord[0]+dataBufferSize[0]
            coordinates[1] = coord[1]+dataBufferSize[1]
            plotArray = cv2.circle(plotArray, coordinates, radius=dotSize, color=plotValuesColour, thickness=-1)
        # Connect Dots if applicable
        if connectDots:
            pass
        return plotArray
    
    def _plotArray_draw_text(plotTitle, plotArray, plotArraySize, dataRange, plotOutlineColour):
        # Text positions/size scale proportionally with plot height.
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontPositionTuningTop = int(.03 * plotArraySize[1])
        fontPositionTuningBot = int(.007 * plotArraySize[1])
        fontSizeTuning = (.001 * plotArraySize[1])
        # putText Max
        textMax = "{}  |  {}".format(round(dataRange[1],4), plotTitle)
        plotArray = cv2.putText(plotArray, textMax, [3, 11 + fontPositionTuningTop], font, .4 + fontSizeTuning, plotOutlineColour, 1)
        # putText Min
        textMin = "{}".format(round(dataRange[0],4))
        plotArray = cv2.putText(plotArray, textMin, [3, plotArraySize[1]-5 - fontPositionTuningBot], font, .4 + fontSizeTuning, plotOutlineColour, 1)
        return plotArray
    
    def _plotArray_draw_text_coords(plotTitle, plotArray, plotArraySize, dataRange, plotOutlineColour):
        # Text positions/size scale proportionally with plot height.
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontPositionTuningTop = int(.03 * plotArraySize[1])
        fontPositionTuningBot = int(.007 * plotArraySize[1])
        fontSizeTuning = (.001 * plotArraySize[1])
        # putText Max
        textMax = "({},{})  |  {}".format(round(dataRange[0][0],4), round(dataRange[0][1],4), plotTitle)
        plotArray = cv2.putText(plotArray, textMax, [3, 11 + fontPositionTuningTop], font, .4 + fontSizeTuning, plotOutlineColour, 1)
        # putText Min
        textMin = "({},{})".format(round(dataRange[1][0],4), round(dataRange[1][1],4))
        plotArray = cv2.putText(plotArray, textMin, [3, plotArraySize[1]-5 - fontPositionTuningBot], font, .4 + fontSizeTuning, plotOutlineColour, 1)
        return plotArray
    
    def _draw_plotArray_to_renderArray(plotArray, renderArray, plotArrayPosition, plotArraySize):
        rH, rW = renderArray.shape[:2]
        r0 = int(plotArrayPosition[1] - (plotArraySize[1] * 0.5))
        r1 = int(plotArrayPosition[1] + (plotArraySize[1] * 0.5))
        c0 = int(plotArrayPosition[0] - (plotArraySize[0] * 0.5))
        c1 = int(plotArrayPosition[0] + (plotArraySize[0] * 0.5))
        # Crop the source plotArray to the portion that falls inside renderArray.
        pr0 = max(-r0, 0)
        pc0 = max(-c0, 0)
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, rH), min(c1, rW)
        if r0 < r1 and c0 < c1:
            renderArray[r0:r1, c0:c1] = plotArray[pr0:pr0+(r1-r0), pc0:pc0+(c1-c0)]
        return renderArray