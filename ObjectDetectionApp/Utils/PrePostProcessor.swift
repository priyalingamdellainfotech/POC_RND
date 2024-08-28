import UIKit

typealias XYXY = (x1: Float, y1: Float, x2: Float, y2: Float)
typealias DisplayColor = UIColor

class IOUCalculator {
    static func IOU(a: XYXY, b: XYXY) -> Float {
        let rectA = CGRect(xyxy: a)
        let rectB = CGRect(xyxy: b)
        return IOU(a: rectA, b: rectB)
    }

    static func IOU(a: CGRect, b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull {
            return 0.0
        }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return Float(intersectionArea / unionArea)
    }
}

struct Prediction {
    let classIndex: Int
    let score: Float
//    let rect: CGRect
    let xyxy: XYXY
    let inputImgSize: CGSize
    
    
}
extension CGRect {
    init(xyxy: XYXY) {
        self.init(
            x: CGFloat(xyxy.x1),
            y: CGFloat(xyxy.y1),
            width: CGFloat(xyxy.x2 - xyxy.x1),
            height: CGFloat(xyxy.y2 - xyxy.y1)
        )
    }
}
class PrePostProcessor: NSObject {
    static let inputWidth = 640
    static let inputHeight = 640

    // Adjusted for YOLOv8: Update outputRow and outputColumn based on the model
    static let outputRow = 8400 // Example value, replace with YOLOv8's output row size
    static let outputColumn = 5 // Example value, replace with YOLOv8's output column size
    static let threshold: Float = 0.35
    static let nmsLimit = 100
    
    // Non-Maximum Suppression (NMS)
    static func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {
        let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
        var selected: [Prediction] = []
        var active = [Bool](repeating: true, count: boxes.count)
        var numActive = active.count
        
        outer: for i in 0..<boxes.count {
            if active[i] {
                let boxA = boxes[sortedIndices[i]]
                selected.append(boxA)
                if selected.count >= limit { break }
                
                for j in i + 1..<boxes.count {
                    if active[j] {
                        let boxB = boxes[sortedIndices[j]]
                        let iouValue = IOUCalculator.IOU(a: boxA.xyxy, b: boxB.xyxy)
                        if iouValue > threshold {
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

//    static func IOU(a: XYXY, b: XYXY) -> Float {
//           let rectA = CGRect(xyxy: a)
//           let rectB = CGRect(xyxy: b)
//           return IOU(a: rectA, b: rectB)
//       }
//    
//    // Intersection Over Union (IOU)
//    static func IOU(a: CGRect, b: CGRect) -> Float {
//        let areaA = a.width * a.height
//        if areaA <= 0 { return 0 }
//        
//        let areaB = b.width * b.height
//        if areaB <= 0 { return 0 }
//        
//        let intersectionMinX = max(a.minX, b.minX)
//        let intersectionMinY = max(a.minY, b.minY)
//        let intersectionMaxX = min(a.maxX, b.maxX)
//        let intersectionMaxY = min(a.maxY, b.maxY)
//        let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) * max(intersectionMaxX - intersectionMinX, 0)
//        return Float(intersectionArea / (areaA + areaB - intersectionArea))
//    }

    static func outputsToNMSPredictions(outputs: [NSNumber], inputSize:CGSize,imageWidth: CGFloat, imageHeight: CGFloat) -> [Prediction] {
        
        let inputSize = inputSize
        
        var predictions = [Prediction]()
        for i in 0..<outputRow {
            
            if Float(truncating: outputs[i * outputColumn + 4]) > threshold {
                let x = Double(truncating: outputs[i * outputColumn])
                let y = Double(truncating: outputs[i * outputColumn + 1])
                let w = Double(truncating: outputs[i * outputColumn + 2])
                let h = Double(truncating: outputs[i * outputColumn + 3])
                
                let left = (x - w / 2)
                let top = (y - h / 2)
                let right = (x + w / 2)
                let bottom = (y + h / 2)
                
                var max = Double(truncating: outputs[i * outputColumn + 5])
                var cls = 0
                for j in 0..<outputColumn - 5 {
                    if Double(truncating: outputs[i * outputColumn + 5 + j]) > max {
                        max = Double(truncating: outputs[i * outputColumn + 5 + j])
                        cls = j
                    }
                }

                let rect = CGRect(x: left, y: top, width: right - left, height: bottom - top).applying(CGAffineTransform(scaleX: imageWidth, y: imageHeight))
          
                
                let prediction = Prediction(
                    classIndex: cls,
                    score: Float(truncating: outputs[i * outputColumn + 4]),
                    xyxy: (
                        x1: Float(rect.origin.x),
                        y1: Float(rect.origin.y),
                        x2: Float(rect.width),
                        y2: Float(rect.height)
                    ),
                    inputImgSize: inputSize
                )

                predictions.append(prediction)
            }
        }

        return nonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: threshold)
    }

    static func cleanDetection(imageView: UIImageView) {
        if let layers = imageView.layer.sublayers {
            for layer in layers {
                if layer is CATextLayer {
                    layer.removeFromSuperlayer()
                }
            }
            for view in imageView.subviews {
                view.removeFromSuperview()
            }
        }
    }

    static func showDetection(imageView: UIImageView, nmsPredictions: [Prediction], classes: [String]) {
        for pred in nmsPredictions {
            let bbox = UIView(frame: CGRect(x: Int(pred.xyxy.x1), y: Int(pred.xyxy.y1), width: Int(pred.xyxy.x2), height: Int(pred.xyxy.y2)))
            bbox.backgroundColor = UIColor.clear
            bbox.layer.borderColor = UIColor.yellow.cgColor
            bbox.layer.borderWidth = 2
            imageView.addSubview(bbox)
            
            let textLayer = CATextLayer()
            textLayer.string = String(format: " %@ %.2f", classes[pred.classIndex], pred.score)
            textLayer.foregroundColor = UIColor.white.cgColor
            textLayer.backgroundColor = UIColor.magenta.cgColor
            textLayer.fontSize = 14
            textLayer.frame = CGRect(x: Int(pred.xyxy.x1), y:Int(pred.xyxy.y1), width: 100, height: 20)
            imageView.layer.addSublayer(textLayer)
        }
    }
}
