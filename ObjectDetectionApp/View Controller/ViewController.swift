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

import UIKit
import AVFoundation
import Vision
import CoreImage
import VideoToolbox

class ViewController: UIViewController {

  // MARK: Storyboards Connections
  @IBOutlet weak var previewView: PreviewView!
  @IBOutlet weak var overlayView: OverlayView!
  @IBOutlet weak var resumeButton: UIButton!
  @IBOutlet weak var cameraUnavailableLabel: UILabel!

  @IBOutlet weak var bottomSheetStateImageView: UIImageView!
  @IBOutlet weak var bottomSheetView: UIView!
  @IBOutlet weak var bottomSheetViewBottomSpace: NSLayoutConstraint!

  // MARK: Constants
    private let displayFont = UIFont.systemFont(ofSize: 14.0, weight: .regular)
  private let edgeOffset: CGFloat = 2.0
  private let labelOffset: CGFloat = 10.0
  private let animationDuration = 0.5
  private let collapseTransitionThreshold: CGFloat = -30.0
  private let expandTransitionThreshold: CGFloat = 30.0
  private let delayBetweenInferencesMs: Double = 200

  // MARK: Instance Variables
  private var initialBottomSpace: CGFloat = 0.0
    let ciContext = CIContext()

  // Holds the results at any time
  private var result: Result?
  private var previousInferenceTimeMs: TimeInterval = Date.distantPast.timeIntervalSince1970 * 100

  // MARK: Controllers that manage functionality
  private lazy var cameraFeedManager = CameraFeedManager(previewView: previewView)
  private var modelDataHandler: ModelDataHandler? =
    ModelDataHandler(modelFileInfo: Yolov5.modelInfo, labelsFileInfo: Yolov5.labelsInfo)
  private var inferenceViewController: InferenceViewController?

    let numberPlateLbl = UILabel()
    var isOCRInProgress = false
    let queue = DispatchQueue(label: "OCRQueue", qos: .userInitiated)

  // MARK: View Handling Methods
  override func viewDidLoad() {
    super.viewDidLoad()

    guard modelDataHandler != nil else {
      fatalError("Failed to load model")
    }
    cameraFeedManager.delegate = self
    overlayView.clearsContextBeforeDrawing = true

    addPanGesture()
      
      // Create the label
      let bottomView = UIView()
      bottomView.translatesAutoresizingMaskIntoConstraints = false
      bottomView.backgroundColor = .lightGray // Set a background color if needed
      view.addSubview(bottomView)
      
      numberPlateLbl.translatesAutoresizingMaskIntoConstraints = false
      numberPlateLbl.text = ""
      numberPlateLbl.textAlignment = .center
      numberPlateLbl.numberOfLines = 0
      bottomView.addSubview(numberPlateLbl)
      
      // Set up constraints for bottomView
      NSLayoutConstraint.activate([
        bottomView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
        bottomView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
        bottomView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        bottomView.heightAnchor.constraint(equalToConstant: 80) // Adjust the height as needed
      ])
      
      // Set up constraints for label
      NSLayoutConstraint.activate([
        numberPlateLbl.centerXAnchor.constraint(equalTo: bottomView.centerXAnchor),
        numberPlateLbl.centerYAnchor.constraint(equalTo: bottomView.centerYAnchor)
      ])
  }

  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    // Dispose of any resources that can be recreated.
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)
    changeBottomViewState()
    cameraFeedManager.checkCameraConfigurationAndStartSession()
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)

    cameraFeedManager.stopSession()
  }

  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }

  // MARK: Button Actions
  @IBAction func onClickResumeButton(_ sender: Any) {

    cameraFeedManager.resumeInterruptedSession { (complete) in

      if complete {
        self.resumeButton.isHidden = true
        self.cameraUnavailableLabel.isHidden = true
      }
      else {
        self.presentUnableToResumeSessionAlert()
      }
    }
  }

  func presentUnableToResumeSessionAlert() {
    let alert = UIAlertController(
      title: "Unable to Resume Session",
      message: "There was an error while attempting to resume session.",
      preferredStyle: .alert
    )
    alert.addAction(UIAlertAction(title: "OK", style: .default, handler: nil))

    self.present(alert, animated: true)
  }

  // MARK: Storyboard Segue Handlers
  override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    super.prepare(for: segue, sender: sender)

    if segue.identifier == "EMBED" {

      guard let tempModelDataHandler = modelDataHandler else {
        return
      }
      inferenceViewController = segue.destination as? InferenceViewController
      inferenceViewController?.wantedInputHeight = tempModelDataHandler.inputHeight
      inferenceViewController?.wantedInputWidth = tempModelDataHandler.inputWidth
      inferenceViewController?.threadCountLimit = tempModelDataHandler.threadCountLimit
      inferenceViewController?.currentThreadCount = tempModelDataHandler.threadCount
      inferenceViewController?.delegate = self

      guard let tempResult = result else {
        return
      }
      inferenceViewController?.inferenceTime = tempResult.inferenceTime

    }
  }
}

// MARK: InferenceViewControllerDelegate Methods
extension ViewController: InferenceViewControllerDelegate {

  func didChangeThreadCount(to count: Int) {
    if modelDataHandler?.threadCount == count { return }
    modelDataHandler = ModelDataHandler(
      modelFileInfo: Yolov5.modelInfo,
      labelsFileInfo: Yolov5.labelsInfo,
      threadCount: count
    )
  }

}

// MARK: CameraFeedManagerDelegate Methods
extension ViewController: CameraFeedManagerDelegate {

    func didOutput(pixelBuffer: CVPixelBuffer, sampleBuffer: CMSampleBuffer,originalImg : UIImage) {
        print("called at \(Date())")
        runModel(onPixelBuffer: pixelBuffer, sampleBuffer: sampleBuffer, originalImg: originalImg)
  }

  // MARK: Session Handling Alerts
  func sessionRunTimeErrorOccurred() {

    // Handles session run time error by updating the UI and providing a button if session can be manually resumed.
    self.resumeButton.isHidden = false
  }

  func sessionWasInterrupted(canResumeManually resumeManually: Bool) {

    // Updates the UI when session is interrupted.
    if resumeManually {
      self.resumeButton.isHidden = false
    }
    else {
      self.cameraUnavailableLabel.isHidden = false
    }
  }

  func sessionInterruptionEnded() {

    // Updates UI once session interruption has ended.
    if !self.cameraUnavailableLabel.isHidden {
      self.cameraUnavailableLabel.isHidden = true
    }

    if !self.resumeButton.isHidden {
      self.resumeButton.isHidden = true
    }
  }

  func presentVideoConfigurationErrorAlert() {

    let alertController = UIAlertController(title: "Configuration Failed", message: "Configuration of camera has failed.", preferredStyle: .alert)
    let okAction = UIAlertAction(title: "OK", style: .cancel, handler: nil)
    alertController.addAction(okAction)

    present(alertController, animated: true, completion: nil)
  }

  func presentCameraPermissionsDeniedAlert() {

    let alertController = UIAlertController(title: "Camera Permissions Denied", message: "Camera permissions have been denied for this app. You can change this by going to Settings", preferredStyle: .alert)

    let cancelAction = UIAlertAction(title: "Cancel", style: .cancel, handler: nil)
    let settingsAction = UIAlertAction(title: "Settings", style: .default) { (action) in

      UIApplication.shared.open(URL(string: UIApplication.openSettingsURLString)!, options: [:], completionHandler: nil)
    }

    alertController.addAction(cancelAction)
    alertController.addAction(settingsAction)

    present(alertController, animated: true, completion: nil)
  }

  /** This method runs the live camera pixelBuffer through tensorFlow to get the result.
   */
    @objc  func runModel(onPixelBuffer pixelBuffer: CVPixelBuffer,sampleBuffer: CMSampleBuffer,originalImg:UIImage) {

    // Run the live camera pixelBuffer through tensorFlow to get the result
    let currentTimeMs = Date().timeIntervalSince1970 * 100

    guard  (currentTimeMs - previousInferenceTimeMs) >= delayBetweenInferencesMs else {
      return
    }

    previousInferenceTimeMs = currentTimeMs
    result = self.modelDataHandler?.runModel2(onFrame: pixelBuffer,sampleBuffer: sampleBuffer)
//      result = self.modelDataHandler?.runModel(onFrame: pixelBuffer)
      
    guard let displayResult = result else {
        DispatchQueue.main.async {
            self.numberPlateLbl.text = ""
            self.overlayView.objectOverlays = []
            self.draw(objectOverlays: self.overlayView.objectOverlays )
        }
       
      return
    }

    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)

    DispatchQueue.main.async {

      // Display results by handing off to the InferenceViewController
//      self.inferenceViewController?.resolution = CGSize(width: width, height: height)

      var inferenceTime: Double = 0
      if let resultInferenceTime = self.result?.inferenceTime {
        inferenceTime = resultInferenceTime
      }
      self.inferenceViewController?.inferenceTime = inferenceTime
      self.inferenceViewController?.tableView.reloadData()

      // Draws the bounding boxes and displays class names and confidence scores.
        self.drawAfterPerformingCalculations(onInferences: displayResult.inferences, withImageSize: CGSize(width: CGFloat(width), height: CGFloat(height)), inputsize: displayResult.inferenceInputImgSize, sampleBuffer: sampleBuffer, originalImg: originalImg)
    }
  }

  /**
   This method takes the results, translates the bounding box rects to the current view, draws the bounding boxes, classNames and confidence scores of inferences.
   */
    func drawAfterPerformingCalculations(onInferences inferences: [Inference], withImageSize imageSize:CGSize,inputsize:CGSize,sampleBuffer : CMSampleBuffer,originalImg : UIImage) {
        
        self.overlayView.objectOverlays = []
        self.overlayView.setNeedsDisplay()
        
        guard !inferences.isEmpty else {
            return
        }
        
        var objectOverlays: [ObjectOverlay] = []
        
        for inference in inferences {
            
            // Translates bounding box rect to current view.
            let x1: CGFloat = CGFloat(inference.xyxy.x1)
            let y1: CGFloat = CGFloat(inference.xyxy.y1)
            let x2: CGFloat = CGFloat(inference.xyxy.x2)
            let y2: CGFloat = CGFloat(inference.xyxy.y2)
            
            let rect = CGRect(
                x: x1,
                y: y2,
                width: x2 - x1,
                height: y1 - y2
            )
            
            var convertedRect = rect.applying(CGAffineTransform(scaleX: self.overlayView.bounds.size.width / inputsize.width , y: self.overlayView.bounds.size.height /  inputsize.height))
            
            print(convertedRect)
            
            //      if convertedRect.origin.x < 0 {
            //        convertedRect.origin.x = self.edgeOffset
            //      }
            //
            //      if convertedRect.origin.y < 0 {
            //        convertedRect.origin.y = self.edgeOffset
            //      }
            //
            //      if convertedRect.maxY > self.overlayView.bounds.maxY {
            //        convertedRect.size.height = self.overlayView.bounds.maxY - convertedRect.origin.y - self.edgeOffset
            //      }
            //
            //      if convertedRect.maxX > self.overlayView.bounds.maxX {
            //        convertedRect.size.width = self.overlayView.bounds.maxX - convertedRect.origin.x - self.edgeOffset
            //      }
            //
            //            CATransaction.begin()
            //            CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
            //        CATransaction.begin()
            //        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
            
            let confidenceValue = Int(inference.confidence * 100.0)
            let string = "(\(confidenceValue)%)"
            
            let size = string.size(usingFont: self.displayFont)
            
            //
            
            let cgrect = CGRect(x: convertedRect.origin.x - 15, y: convertedRect.origin.y, width: convertedRect.width + 20, height: convertedRect.height)
            
            
            let objectOverlay = ObjectOverlay(name: string, borderRect: cgrect, nameStringSize: size, color: UIColor.red, font: self.displayFont)
            //objectOverlays.append(objectOverlay)
            
            let cgrect1 = CGRect(x: convertedRect.origin.x  , y: convertedRect.origin.y , width: convertedRect.width + 40, height: convertedRect.height + 40)
//            if confidenceValue > 70
//            {
                if let croppedImage = originalImg.cropped(to: cgrect1) {
                    if #available(iOS 13.0, *) {
                        performOCROnImage(croppedImage) { recognizedText in
                            if let text = recognizedText {
                                
                                DispatchQueue.main.async
                                {
                                    self.numberPlateLbl.text = text
                                }
                                
                                print("Recognized Text: \(text)")
                            } else {
                                print("No text recognized")
                            }
                        }
                    } else {
                        // Fallback on earlier versions
                        print("OCR requires iOS 13.0 or later")
                    }
                } else {
                    print("Failed to crop image")
                }
                objectOverlays.append(objectOverlay)
           // }
        }
        // Hands off drawing to the OverlayView
        self.draw(objectOverlays: objectOverlays)
        //CATransaction.commit()
    }
    
    @available(iOS 13.0, *)
    func performOCROnImage(_ image: UIImage, completion: @escaping (String?) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(nil)
            return
        }

        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        let request = VNRecognizeTextRequest { (request, error) in
            guard error == nil else {
                completion(nil)
                return
            }

            // Define regex pattern for number plates (adjust as needed)
            // Example pattern for alphanumeric number plates with spaces or dashes (adjust based on your region)
            let numberPlatePattern = "[A-Z0-9]{2,3}-?[A-Z0-9]{2,4}"

            let recognizedStrings = request.results?.compactMap { result in
                (result as? VNRecognizedTextObservation)?.topCandidates(1).first?.string
            }

            // Filter recognized text using the regex pattern
            let numberPlates = recognizedStrings?.filter { text in
                let range = NSRange(location: 0, length: text.utf16.count)
                let regex = try? NSRegularExpression(pattern: numberPlatePattern)
                return regex?.firstMatch(in: text, options: [], range: range) != nil
            }

            completion(numberPlates?.joined(separator: "\n"))
        }

        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        do {
            try requestHandler.perform([request])
        } catch {
            completion(nil)
        }
    }


    
    
//   func performOCROnImage(_ image: UIImage, completion: @escaping (String?) -> Void) {
//           guard let cgImage = image.cgImage else {
//               completion(nil)
//               return
//           }
//    
//           let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
//           let request = VNRecognizeTextRequest { (request, error) in
//               guard error == nil else {
//                   completion(nil)
//                   return
//               }
//    
//               // Define regex pattern for number plates
//               let numberPlatePattern = "[A-Z0-9]{2,3}-?[A-Z0-9]{2,4}"
//               let recognizedStrings = request.results?.compactMap { result in
//                   (result as? VNRecognizedTextObservation)?.topCandidates(1).first?.string
//               }
//    
//               // Filter recognized text using the regex pattern
//               let numberPlates = recognizedStrings?.filter { text in
//                   let range = NSRange(location: 0, length: text.utf16.count)
//                   let regex = try? NSRegularExpression(pattern: numberPlatePattern)
//                   return regex?.firstMatch(in: text, options: [], range: range) != nil
//               }
//    
//               var anotherArray: [String] = []
//               numberPlates?.joined(separator: " ")
//               anotherArray = numberPlates.filter { containsDigits($0) }
//               
//               completion(anotherArray)
//           }
//    
//           request.recognitionLevel = .accurate
//           request.usesLanguageCorrection = true
//    
//           do {
//               try requestHandler.perform([request])
//           } catch {
//               completion(nil)
//           }
//       }
    
    func containsDigits(_ string: String) -> Bool {
        return string.rangeOfCharacter(from: CharacterSet.decimalDigits) != nil
    }
     
     




  /** Calls methods to update overlay view with detected bounding boxes and class names.
   */
  func draw(objectOverlays: [ObjectOverlay]) {
      self.overlayView.objectOverlays = []
//      self.draw(objectOverlays: self.overlayView.objectOverlays )
    self.overlayView.objectOverlays = objectOverlays
      DispatchQueue.main.async {
          self.overlayView.setNeedsDisplay()
      }
  }
 
}

// MARK: Bottom Sheet Interaction Methods
extension ViewController {

  // MARK: Bottom Sheet Interaction Methods
  /**
   This method adds a pan gesture to make the bottom sheet interactive.
   */
  private func addPanGesture() {
    let panGesture = UIPanGestureRecognizer(target: self, action: #selector(ViewController.didPan(panGesture:)))
    bottomSheetView.addGestureRecognizer(panGesture)
  }


  /** Change whether bottom sheet should be in expanded or collapsed state.
   */
  private func changeBottomViewState() {

    guard let inferenceVC = inferenceViewController else {
      return
    }

    if bottomSheetViewBottomSpace.constant == inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height {

      bottomSheetViewBottomSpace.constant = 0.0
    }
    else {
      bottomSheetViewBottomSpace.constant = inferenceVC.collapsedHeight - bottomSheetView.bounds.size.height
    }
    setImageBasedOnBottomViewState()
  }

  /**
   Set image of the bottom sheet icon based on whether it is expanded or collapsed
   */
  private func setImageBasedOnBottomViewState() {

    if bottomSheetViewBottomSpace.constant == 0.0 {
      bottomSheetStateImageView.image = UIImage(named: "down_icon")
    }
    else {
      bottomSheetStateImageView.image = UIImage(named: "up_icon")
    }
  }
    
    func imageFromSampleBufferUsingVideoToolbox(sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return nil
        }

        var cgImage: CGImage?
        let status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)
        
        guard status == kCVReturnSuccess, let createdImage = cgImage else {
            return nil
        }
        
        return UIImage(cgImage: createdImage)
    }


  /**
   This method responds to the user panning on the bottom sheet.
   */
  @objc func didPan(panGesture: UIPanGestureRecognizer) {

    // Opens or closes the bottom sheet based on the user's interaction with the bottom sheet.
    let translation = panGesture.translation(in: view)

    switch panGesture.state {
    case .began:
      initialBottomSpace = bottomSheetViewBottomSpace.constant
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .changed:
      translateBottomSheet(withVerticalTranslation: translation.y)
    case .cancelled:
      setBottomSheetLayout(withBottomSpace: initialBottomSpace)
    case .ended:
      translateBottomSheetAtEndOfPan(withVerticalTranslation: translation.y)
      setImageBasedOnBottomViewState()
      initialBottomSpace = 0.0
    default:
      break
    }
  }

  /**
   This method sets bottom sheet translation while pan gesture state is continuously changing.
   */
  private func translateBottomSheet(withVerticalTranslation verticalTranslation: CGFloat) {

    let bottomSpace = initialBottomSpace - verticalTranslation
    guard bottomSpace <= 0.0 && bottomSpace >= inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height else {
      return
    }
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   This method changes bottom sheet state to either fully expanded or closed at the end of pan.
   */
  private func translateBottomSheetAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) {

    // Changes bottom sheet state to either fully open or closed at the end of pan.
    let bottomSpace = bottomSpaceAtEndOfPan(withVerticalTranslation: verticalTranslation)
    setBottomSheetLayout(withBottomSpace: bottomSpace)
  }

  /**
   Return the final state of the bottom sheet view (whether fully collapsed or expanded) that is to be retained.
   */
  private func bottomSpaceAtEndOfPan(withVerticalTranslation verticalTranslation: CGFloat) -> CGFloat {

    // Calculates whether to fully expand or collapse bottom sheet when pan gesture ends.
    var bottomSpace = initialBottomSpace - verticalTranslation

    var height: CGFloat = 0.0
    if initialBottomSpace == 0.0 {
      height = bottomSheetView.bounds.size.height
    }
    else {
      height = inferenceViewController!.collapsedHeight
    }

    let currentHeight = bottomSheetView.bounds.size.height + bottomSpace

    if currentHeight - height <= collapseTransitionThreshold {
      bottomSpace = inferenceViewController!.collapsedHeight - bottomSheetView.bounds.size.height
    }
    else if currentHeight - height >= expandTransitionThreshold {
      bottomSpace = 0.0
    }
    else {
      bottomSpace = initialBottomSpace
    }

    return bottomSpace
  }

  /**
   This method layouts the change of the bottom space of bottom sheet with respect to the view managed by this controller.
   */
  func setBottomSheetLayout(withBottomSpace bottomSpace: CGFloat) {

    view.setNeedsLayout()
    bottomSheetViewBottomSpace.constant = bottomSpace
    view.setNeedsLayout()
  }

}

extension UIImage {
    func cropped(to rect: CGRect) -> UIImage? {
        guard let cgImage = self.cgImage else {
            return nil
        }
        
        // Adjust rect to consider image scale
        let scaledRect = CGRect(x: rect.origin.x * self.scale,
                                y: rect.origin.y * self.scale,
                                width: rect.size.width * self.scale,
                                height: rect.size.height * self.scale)
        
        // Perform cropping using Core Graphics
        guard let croppedCGImage = cgImage.cropping(to: scaledRect) else {
            return nil
        }
        
        // Create a new UIImage from the cropped CGImage
        let croppedImage = UIImage(cgImage: croppedCGImage, scale: self.scale, orientation: self.imageOrientation)
        
        return croppedImage
    }
}
