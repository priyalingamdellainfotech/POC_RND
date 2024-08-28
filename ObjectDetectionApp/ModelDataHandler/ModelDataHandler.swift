// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit
import Accelerate
import UIKit
import AVFoundation

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferenceTime: Double
    let inferences: [Inference]
    let inferenceInputImgSize: CGSize
}

/// Stores one formatted inference.
struct Inference {
    let confidence: Float
    let className: String
//    let rect: CGRect
    let xyxy: XYXY
    let displayColor: DisplayColor
    
    static var redColor: DisplayColor {
            return UIColor.red
        }
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum Yolov5 {
    static let modelInfo: FileInfo = (name: "", extension: "tflite")
    static let labelsInfo: FileInfo = (name: "classes", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10
    
    let threshold: Float = 0.60
    
    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640
    let inputHeight = 640
    
    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5
    
    // MARK: Private properties
    private var labels: [String] = []
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    // MARK: - Initialization
    
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo, threadCount: Int = 1) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
        
        // Load the classes listed in the labels file.
        loadLabels(fileInfo: labelsFileInfo)
    }
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
        let interval: TimeInterval
        let outputResult: Tensor

        do {
            let inputTensor = try interpreter.input(at: 0)
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            interval = Date().timeIntervalSince(startDate) * 1000
            
            outputResult = try interpreter.output(at: 0)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        print("outputResult\(outputResult)")
        
        let outputs = ([Float](unsafeData: outputResult.data) ?? []) as [NSNumber]
        
        let input: Tensor
        do {
            input = try interpreter.input(at: 0)
        } catch let error {
            NSLog("Failed to get input with error: \(error.localizedDescription)")
            return nil
        }
        
        let inputSize = CGSize(
            width: input.shape.dimensions[1],
            height: input.shape.dimensions[2]
        )

        let nmsPredictions = PrePostProcessor.outputsToNMSPredictions(outputs: outputs, inputSize: inputSize, imageWidth: CGFloat(imageWidth), imageHeight: CGFloat(imageHeight))
print("nmsPredictions\(nmsPredictions)")
        var inference: [Inference] = []
        for prediction in nmsPredictions {
            guard prediction.score >= threshold else {
                continue
            }
            let pred = Inference(confidence: prediction.score, className: labels[prediction.classIndex], xyxy: prediction.xyxy, displayColor: colorForClass(withIndex: prediction.classIndex + 1))
            inference.append(pred)
        }
        let result = Result(inferenceTime: interval, inferences: inference, inferenceInputImgSize: inputSize)

        return result
    }
    
     func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage)
    }
 
    
    func runModel2(onFrame pixelBuffer: CVPixelBuffer,sampleBuffer: CMSampleBuffer) -> Result? {
        
        NSLog("Start inference using TFLite")
        
        let modelFilePath = Bundle.main.url(
//            coco128-yolov8n-seg_float16
            forResource: "best_float16",
            withExtension: "tflite")!.path
        
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
//        let imageChannels = 4
//        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
//        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
//        
//        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
//            return nil
//        }
        
        let interpreter: Interpreter
        do {
            interpreter = try Interpreter(
                modelPath: modelFilePath,
                delegates: []
            )
            
            try interpreter.allocateTensors()
        } catch {
            NSLog("Error while initializing interpreter: \(error)")
           return nil
        }
        
        let input: Tensor
        do {
            input = try interpreter.input(at: 0)
        } catch let error {
            NSLog("Failed to get input with error: \(error.localizedDescription)")
            return nil
        }
        
        
        
        let inputSize = CGSize(
            width: input.shape.dimensions[1],
            height: input.shape.dimensions[2]
        )
        
        let uiImage = imageFromSampleBuffer(sampleBuffer: sampleBuffer )
        guard let data = uiImage!.resized(to: inputSize).normalizedDataFromImage() else {
            return nil
        }
        
        let boxesOutputTensor: Tensor
//            let masksOutputTensor: Tensor
        
        do {
            try interpreter.copy(data, toInputAt: 0)
//            await setStatus(to: .inferencing)
            try interpreter.invoke()
//            await setStatus(to: .postProcessing)
            
            boxesOutputTensor = try interpreter.output(at: 0)
//                masksOutputTensor = try interpreter.output(at: 1)
        } catch let error {
            NSLog("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        let boxesOutputShapeDim = boxesOutputTensor.shape.dimensions
//            let masksOutputShapeDim = masksOutputTensor.shape.dimensions
        
        NSLog("Got output with index 0 (boxes) with shape: \(boxesOutputShapeDim)")
//            NSLog("Got output with index 1 (masks) with shape: \(masksOutputShapeDim)")
        
        let numSegmentationMasks = 0
        let numClasses = boxesOutputShapeDim[1] - 4 - numSegmentationMasks
        
        NSLog("Model has \(numClasses) classes")
        
        let boxesOutput = ([Float](unsafeData: boxesOutputTensor.data) ?? [])
        
        
        let interval: TimeInterval
        let startDate = Date()
        interval = Date().timeIntervalSince(startDate) * 100
        
        var predictions = getPredictionsFromOutput(
            output: boxesOutput as [NSNumber],
            rows: boxesOutputShapeDim[1],
            columns: boxesOutputShapeDim[2],
            numberOfClasses: numClasses,
            inputImgSize: inputSize
        )

        NSLog("Got \(predictions.count) predicted boxes")
        NSLog("Remove predictions with score lower than 0.3")
        var inference: [Inference] = []
        predictions.removeAll { $0.score < threshold }
        
        NSLog("\(predictions.count) predicted boxes left after removing predictions with score lower than 0.3")
        
        guard !predictions.isEmpty else {
            return nil
        }
        
        NSLog("Perform non maximum suppression")
        
        // Group predictions by class
        let groupedPredictions = Dictionary(grouping: predictions) { prediction in
            prediction.classIndex
        }
        
        
        var nmsPredictionsValue: [Prediction] = []
        let _ = groupedPredictions.mapValues { predictions in
            nmsPredictionsValue.append(
                contentsOf: nonMaximumSuppression(
                    predictions: predictions,
                    iouThreshold: 0.6,
                    limit: 100))
        }
        
        NSLog("\(predictions.count) boxes left after performing nms with iou threshold of 0.6")
        
        guard !predictions.isEmpty else {
            return nil
        }
        
        predictions = predictions.map { prediction in
            return Prediction(
                classIndex: prediction.classIndex,
                score: prediction.score,
                xyxy: (
                    prediction.xyxy.x1 * Float(inputSize.width),
                    prediction.xyxy.y1 * Float(inputSize.height),
                    prediction.xyxy.x2 * Float(inputSize.width),
                    prediction.xyxy.y2 * Float(inputSize.height)
                ),
//                    maskCoefficients: prediction.maskCoefficients,
                inputImgSize: prediction.inputImgSize)
        }
        
        
        print("final prediction\(predictions)")
        
        for prediction in predictions {
         
            let pred = Inference(confidence: prediction.score, className: "class name", xyxy: prediction.xyxy, displayColor: UIColor.red)
            inference.append(pred)
            
        }
        let result = Result(inferenceTime: interval, inferences: inference,inferenceInputImgSize: inputSize)
        inference.removeAll()

        return result
    }
    
    
    func getPredictionsFromOutput(
        output: [NSNumber],
        rows: Int,
        columns: Int,
        numberOfClasses: Int,
        inputImgSize: CGSize
    ) -> [Prediction] {
        guard !output.isEmpty else {
            return []
        }
        var predictions = [Prediction]()
        for i in 0..<columns {
            let centerX = Float(truncating: output[0*columns+i])
            let centerY = Float(truncating: output[1*columns+i])
            let width   = Float(truncating: output[2*columns+i])
            let height  = Float(truncating: output[3*columns+i])
            
            let (classIndex, score) = {
                var classIndex: Int = 0
                var heighestScore: Float = 0
                for j in 0..<numberOfClasses {
                    let score = Float(truncating: output[(4+j)*columns+i])
                    if score > heighestScore {
                        heighestScore = score
                        classIndex = j
                    }
                }
                return (classIndex, heighestScore)
            }()
            
//            let maskCoefficients = {
//                var coefficients: [Float] = []
//                for k in 0..<32 {
//                    coefficients.append(Float(truncating: output[(4+numberOfClasses+k)*columns+i]))
//                }
//                return coefficients
//            }()
            
            // Convert box from xywh to xyxy
            let left = centerX - width/2
            let top = centerY - height/2
            let right = centerX + width/2
            let bottom = centerY + height/2
            
            let prediction = Prediction(
                classIndex: classIndex,
                score: score,
                xyxy: (left, top, right, bottom),
//                maskCoefficients: maskCoefficients,
                inputImgSize: inputImgSize
            )
            predictions.append(prediction)
        }
        
        return predictions
    }
    
    
    func nonMaximumSuppression(
        predictions: [Prediction],
        iouThreshold: Float,
        limit: Int
    ) -> [Prediction] {
        guard !predictions.isEmpty else {
            return []
        }
        
        let sortedIndices = predictions.indices.sorted {
            predictions[$0].score > predictions[$1].score
        }
        
        var selected: [Prediction] = []
        var active = [Bool](repeating: true, count: predictions.count)
        var numActive = active.count

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        outer: for i in 0..<predictions.count {
            
            if active[i] {
                
                let boxA = predictions[sortedIndices[i]]
                selected.append(boxA)
                
                if selected.count >= limit { break }

                for j in i+1..<predictions.count {
                
                    if active[j] {
                
                        let boxB = predictions[sortedIndices[j]]
                        
                        if IOU(a: boxA.xyxy, b: boxB.xyxy) > iouThreshold {
                            
                            active[j] = false
                            numActive -= 1
                           
                            if numActive <= 0 { break outer }
                        
                        }
                    
                    }
                
                }
            }
            
        }
        return selected
    }
    
    private func IOU(a: XYXY, b: XYXY) -> Float {
        // Calculate the intersection coordinates
        let x1 = max(a.x1, b.x1)
        let y1 = max(a.y1, b.y1)
        let x2 = max(a.x2, b.x2)
        let y2 = max(a.y1, b.y2)
        
        // Calculate the intersection area
        let intersection = max(x2 - x1, 0) * max(y2 - y1, 0)
        
        // Calculate the union area
        let area1 = (a.x2 - a.x1) * (a.y2 - a.y1)
        let area2 = (b.x2 - b.x1) * (b.y2 - b.y1)
        let union = area1 + area2 - intersection
        
        // Calculate the IoU score
        let iou = intersection / union
        
        return iou
    }


    /// Filters out all the results with confidence score < threshold and returns the top N results
    /// sorted in descending order.
/*    func formatResults(boundingBox: [Float], outputClasses: [Float], outputScores: [Float], outputCount: Int, width: CGFloat, height: CGFloat) -> [Inference]{
        var resultsArray: [Inference] = []
        if (outputCount == 0) {
            return resultsArray
        }
        for i in 0...outputCount - 1 {
            
            let score = outputScores[i]
            
            // Filters results with confidence < threshold.
            guard score >= threshold else {
                continue
            }
            
            // Gets the output class names for detected classes from labels list.
            let outputClassIndex = Int(outputClasses[i])
            let outputClass = labels[outputClassIndex + 1]
            
            var rect: CGRect = CGRect.zero
            
            // Translates the detected bounding box to CGRect.
            rect.origin.y = CGFloat(boundingBox[4*i])
            rect.origin.x = CGFloat(boundingBox[4*i+1])
            rect.size.height = CGFloat(boundingBox[4*i+2]) - rect.origin.y
            rect.size.width = CGFloat(boundingBox[4*i+3]) - rect.origin.x
            
            // The detected corners are for model dimensions. So we scale the rect with respect to the
            // actual image dimensions.
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))
            
            // Gets the color assigned for the class
            let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
            let inference = Inference(confidence: score,
                                      className: outputClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }
        
        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.confidence  > second.confidence
        }
        
        return resultsArray
    }*/
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
           var labels = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                       "valid labels file and try again.")
        }
    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append((Float(bytes[i]) - imageMean) / imageStd)
        }
        return Data(copyingBufferOf: floats)
    }
    
    /// This assigns color for a particular class.
    private func colorForClass(withIndex index: Int) -> UIColor {
        
        // We have a set of colors and the depending upon a stride, it assigns variations to of the base
        // colors to each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}
