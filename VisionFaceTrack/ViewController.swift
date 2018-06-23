/*
 See LICENSE folder for this sample’s licensing information.
 
 Abstract:
 Contains the main app implementation using Vision.
 */

import UIKit
import AVKit
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // Main view for showing camera content.
    @IBOutlet weak var previewView: UIView?
    @IBOutlet weak var infoLabel: UILabel!
    @IBOutlet weak var blurView: UIVisualEffectView!
    
    // AVCapture variables to hold sequence data
    var session: AVCaptureSession?
    var previewLayer: AVCaptureVideoPreviewLayer?
    
    var videoDataOutput: AVCaptureVideoDataOutput?
    var videoDataOutputQueue: DispatchQueue?
    
    var captureDevice: AVCaptureDevice?
    var captureDeviceResolution: CGSize = CGSize()
    let modelCNN = SmileCNN()
    let modelNet = SmileNet()
    var captureDevicePosition = AVCaptureDevice.Position.front
    
    // Layer UI for drawing Vision results
    var rootLayer: CALayer?
    var detectionOverlayLayer: CALayer?
    var detectedFaceRectangleShapeLayer: CAShapeLayer?
    var detectedFaceLandmarksShapeLayer: CAShapeLayer?
    
    // Vision requests
    private var detectionRequests: [VNDetectFaceRectanglesRequest]?
    private var trackingRequests: [VNTrackObjectRequest]?
    
    lazy var sequenceRequestHandler = VNSequenceRequestHandler()
    
    var backcam = false
    
    // MARK: UIViewController overrides
    
    override func viewDidLoad() {
        super.viewDidLoad()
        blurView.layer.cornerRadius = 10
        blurView.clipsToBounds = true
        
        self.session = self.setupAVCaptureSession()
        
        self.prepareVisionRequest()
        
        self.session?.startRunning()
    }
    
    @IBAction func changeCameraView(_ sender: UISwitch) {
        captureDevicePosition = sender.isOn ? .front : .back
        if (captureDevicePosition == .back){
            backcam = true
        }else{
            backcam = false
        }
        self.infoLabel.text = "Find Faces"
        self.session?.stopRunning()
        
        self.session = self.setupAVCaptureSession()
        self.prepareVisionRequest()
        self.session?.startRunning()
    }
    // Ensure that the interface stays locked in Portrait.
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask {
        return .portrait
    }
    
    // Ensure that the interface stays locked in Portrait.
    override var preferredInterfaceOrientationForPresentation: UIInterfaceOrientation {
        return .portrait
    }
    
    // MARK: AVCapture Setup
    
    /// - Tag: CreateCaptureSession
    fileprivate func setupAVCaptureSession() -> AVCaptureSession? {
        let captureSession = AVCaptureSession()
        do {
            let inputDevice = try self.configureFrontCamera(for: captureSession)
            self.configureVideoDataOutput(for: inputDevice.device, resolution: inputDevice.resolution, captureSession: captureSession)
            self.designatePreviewLayer(for: captureSession)
            return captureSession
        } catch let executionError as NSError {
            self.presentError(executionError)
        } catch {
            self.presentErrorAlert(message: "An unexpected failure has occured")
        }
        
        self.teardownAVCapture()
        
        return nil
    }
    
    /// - Tag: ConfigureDeviceResolution
    fileprivate func highestResolution420Format(for device: AVCaptureDevice) -> (format: AVCaptureDevice.Format, resolution: CGSize)? {
        var highestResolutionFormat: AVCaptureDevice.Format? = nil
        var highestResolutionDimensions = CMVideoDimensions(width: 0, height: 0)
        
        for format in device.formats {
            let deviceFormat = format as AVCaptureDevice.Format
            
            let deviceFormatDescription = deviceFormat.formatDescription
            if deviceFormatDescription.mediaSubType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange {
                let candidateDimensions = deviceFormatDescription.videoDimensions
                if (highestResolutionFormat == nil) || (candidateDimensions.width > highestResolutionDimensions.width) {
                    highestResolutionFormat = deviceFormat
                    highestResolutionDimensions = candidateDimensions
                }
            }
        }
        
        if highestResolutionFormat != nil {
            let resolution = CGSize(width: CGFloat(highestResolutionDimensions.width), height: CGFloat(highestResolutionDimensions.height))
            return (highestResolutionFormat!, resolution)
        }
        
        return nil
    }
    
    fileprivate func configureFrontCamera(for captureSession: AVCaptureSession) throws -> (device: AVCaptureDevice, resolution: CGSize) {
        let deviceDiscoverySession = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: captureDevicePosition)
        
        if let device = deviceDiscoverySession.devices.first {
            if let deviceInput = try? AVCaptureDeviceInput(device: device) {
                if captureSession.canAddInput(deviceInput) {
                    captureSession.addInput(deviceInput)
                }
                
                if let highestResolution = self.highestResolution420Format(for: device) {
                    try device.lockForConfiguration()
                    device.activeFormat = highestResolution.format
                    device.unlockForConfiguration()
                    
                    return (device, highestResolution.resolution)
                }
            }
        }
        
        throw NSError(domain: "ViewController", code: 1, userInfo: nil)
    }
    
    /// - Tag: CreateSerialDispatchQueue
    fileprivate func configureVideoDataOutput(for inputDevice: AVCaptureDevice, resolution: CGSize, captureSession: AVCaptureSession) {
        
        let videoDataOutput = AVCaptureVideoDataOutput()
        videoDataOutput.alwaysDiscardsLateVideoFrames = true
        
        // Create a serial dispatch queue used for the sample buffer delegate as well as when a still image is captured.
        // A serial dispatch queue must be used to guarantee that video frames will be delivered in order.
        let videoDataOutputQueue = DispatchQueue(label: "com.example.apple-samplecode.VisionFaceTrack")
        videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        
        if captureSession.canAddOutput(videoDataOutput) {
            captureSession.addOutput(videoDataOutput)
        }
        
        videoDataOutput.connection(with: .video)?.isEnabled = true
        
        if let captureConnection = videoDataOutput.connection(with: AVMediaType.video) {
            if captureConnection.isCameraIntrinsicMatrixDeliverySupported {
                captureConnection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }
        
        self.videoDataOutput = videoDataOutput
        self.videoDataOutputQueue = videoDataOutputQueue
        
        self.captureDevice = inputDevice
        self.captureDeviceResolution = resolution
    }
    
    /// - Tag: DesignatePreviewLayer
    fileprivate func designatePreviewLayer(for captureSession: AVCaptureSession) {
        let videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        self.previewLayer = videoPreviewLayer
        
        videoPreviewLayer.name = "CameraPreview"
        videoPreviewLayer.backgroundColor = UIColor.black.cgColor
        videoPreviewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        
        if let previewRootLayer = self.previewView?.layer {
            self.rootLayer = previewRootLayer
            
            previewRootLayer.masksToBounds = true
            videoPreviewLayer.frame = previewRootLayer.bounds
            previewRootLayer.addSublayer(videoPreviewLayer)
        }
    }
    
    // Removes infrastructure for AVCapture as part of cleanup.
    fileprivate func teardownAVCapture() {
        self.videoDataOutput = nil
        self.videoDataOutputQueue = nil
        
        if let previewLayer = self.previewLayer {
            previewLayer.removeFromSuperlayer()
            self.previewLayer = nil
        }
    }
    
    // MARK: Helper Methods for Error Presentation
    
    fileprivate func presentErrorAlert(withTitle title: String = "Unexpected Failure", message: String) {
        let alertController = UIAlertController(title: title, message: message, preferredStyle: .alert)
        self.present(alertController, animated: true)
    }
    
    fileprivate func presentError(_ error: NSError) {
        self.presentErrorAlert(withTitle: "Failed with error \(error.code)", message: error.localizedDescription)
    }
    
    // MARK: Helper Methods for Handling Device Orientation & EXIF
    
    fileprivate func radiansForDegrees(_ degrees: CGFloat) -> CGFloat {
        return CGFloat(Double(degrees) * Double.pi / 180.0)
    }
    
    func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        switch deviceOrientation {
        case .portraitUpsideDown:
            return .rightMirrored

        case .landscapeLeft:
            return .downMirrored

        case .landscapeRight:
            return .upMirrored

        default:
            return .leftMirrored
        }
    }
    
    func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
        return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
    }
    
    // MARK: Performing Vision Requests
    
    /// - Tag: WriteCompletionHandler
    fileprivate func prepareVisionRequest() {
        
        self.trackingRequests = []
        var requests = [VNTrackObjectRequest]()
        
        let faceDetectionRequest = VNDetectFaceRectanglesRequest(completionHandler: { (request, error) in
            
            if error != nil {
                print("FaceDetection error: \(String(describing: error)).")
            }
            
            guard let faceDetectionRequest = request as? VNDetectFaceRectanglesRequest,
                let results = faceDetectionRequest.results as? [VNFaceObservation] else {
                    return
            }
            DispatchQueue.main.async {
                // Add the observations to the tracking list
                for observation in results {
                    let faceTrackingRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                    requests.append(faceTrackingRequest)
                    
                }
                self.trackingRequests = requests
            }
        })
        
        // Start with detection.  Find face, then track it.
        self.detectionRequests = [faceDetectionRequest]
        
        self.sequenceRequestHandler = VNSequenceRequestHandler()
        
        self.setupVisionDrawingLayers()
    }
    
    // MARK: Drawing Vision Observations
    
    fileprivate func setupVisionDrawingLayers() {
        let captureDeviceResolution = self.captureDeviceResolution
        
        let captureDeviceBounds = CGRect(x: 0,
                                         y: 0,
                                         width: captureDeviceResolution.width,
                                         height: captureDeviceResolution.height)
        
        let captureDeviceBoundsCenterPoint = CGPoint(x: captureDeviceBounds.midX,
                                                     y: captureDeviceBounds.midY)
        
        let normalizedCenterPoint = CGPoint(x: 0.5, y: 0.5)
        
        guard let rootLayer = self.rootLayer else {
            self.presentErrorAlert(message: "view was not property initialized")
            return
        }
        
        let overlayLayer = CALayer()
        overlayLayer.name = "DetectionOverlay"
        overlayLayer.masksToBounds = true
        overlayLayer.anchorPoint = normalizedCenterPoint
        overlayLayer.bounds = captureDeviceBounds
        overlayLayer.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        
        let faceRectangleShapeLayer = CAShapeLayer()
        faceRectangleShapeLayer.name = "RectangleOutlineLayer"
        faceRectangleShapeLayer.bounds = captureDeviceBounds
        faceRectangleShapeLayer.anchorPoint = normalizedCenterPoint
        faceRectangleShapeLayer.position = captureDeviceBoundsCenterPoint
        faceRectangleShapeLayer.fillColor = nil
        faceRectangleShapeLayer.strokeColor = UIColor.green.withAlphaComponent(0.7).cgColor
        faceRectangleShapeLayer.lineWidth = 5
        faceRectangleShapeLayer.shadowOpacity = 0.7
        faceRectangleShapeLayer.shadowRadius = 5
        
        let faceLandmarksShapeLayer = CAShapeLayer()
        faceLandmarksShapeLayer.name = "FaceLandmarksLayer"
        faceLandmarksShapeLayer.bounds = captureDeviceBounds
        faceLandmarksShapeLayer.anchorPoint = normalizedCenterPoint
        faceLandmarksShapeLayer.position = captureDeviceBoundsCenterPoint
        faceLandmarksShapeLayer.fillColor = nil
        faceLandmarksShapeLayer.strokeColor = UIColor.yellow.withAlphaComponent(0.7).cgColor
        faceLandmarksShapeLayer.lineWidth = 3
        faceLandmarksShapeLayer.shadowOpacity = 0.7
        faceLandmarksShapeLayer.shadowRadius = 5
        
        overlayLayer.addSublayer(faceRectangleShapeLayer)
        faceRectangleShapeLayer.addSublayer(faceLandmarksShapeLayer)
        rootLayer.addSublayer(overlayLayer)
        
        self.detectionOverlayLayer = overlayLayer
        self.detectedFaceRectangleShapeLayer = faceRectangleShapeLayer
        self.detectedFaceLandmarksShapeLayer = faceLandmarksShapeLayer
        
        self.updateLayerGeometry()
    }
    
    
    fileprivate func updateLayerGeometry() {
        guard let overlayLayer = self.detectionOverlayLayer,
            let rootLayer = self.rootLayer,
            let previewLayer = self.previewLayer
            else {
                return
        }
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let videoPreviewRect = previewLayer.layerRectConverted(fromMetadataOutputRect: CGRect(x: 0, y: 0, width: 1, height: 1))
        
        var rotation: CGFloat
        var scaleX: CGFloat
        var scaleY: CGFloat
        
        // Rotate the layer into screen orientation.
        switch captureDevicePosition {
        case .front:
            switch UIDevice.current.orientation {
            case .portraitUpsideDown:
                rotation = 180
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
                
            case .landscapeLeft:
                rotation = 90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
                
            case .landscapeRight:
                rotation = -90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
                
            default:
                rotation = 0
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
            }
        case .back:
            switch UIDevice.current.orientation {
            case .portraitUpsideDown:
                rotation = 180
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
                
            case .landscapeLeft:
                rotation = 90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
                
            case .landscapeRight:
                rotation = -90
                scaleX = videoPreviewRect.height / captureDeviceResolution.width
                scaleY = scaleX
                
            default:
                rotation = 0
                scaleX = videoPreviewRect.width / captureDeviceResolution.width
                scaleY = videoPreviewRect.height / captureDeviceResolution.height
            }
        default:
            return
        }

        // Scale and mirror the image to ensure upright presentation.
        if backcam{
            let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation))
                .scaledBy(x: -scaleX, y: -scaleY)
            overlayLayer.setAffineTransform(affineTransform)
        }else{
            let affineTransform = CGAffineTransform(rotationAngle: radiansForDegrees(rotation))
                .scaledBy(x: scaleX, y: -scaleY)
            overlayLayer.setAffineTransform(affineTransform)
        }
        
        
        // Cover entire screen UI.
        let rootLayerBounds = rootLayer.bounds
        overlayLayer.position = CGPoint(x: rootLayerBounds.midX, y: rootLayerBounds.midY)
    }
    
    fileprivate func addPoints(in landmarkRegion: VNFaceLandmarkRegion2D, to path: CGMutablePath, applying affineTransform: CGAffineTransform, closingWhenComplete closePath: Bool) {
        let pointCount = landmarkRegion.pointCount
        if pointCount > 1 {
            let points: [CGPoint] = landmarkRegion.normalizedPoints
            path.move(to: points[0], transform: affineTransform)
            path.addLines(between: points, transform: affineTransform)
            if closePath {
                path.addLine(to: points[0], transform: affineTransform)
                path.closeSubpath()
            }
        }
    }
    
    fileprivate func addIndicators(to faceRectanglePath: CGMutablePath, faceLandmarksPath: CGMutablePath, for faceObservation: VNFaceObservation) {
        let displaySize = self.captureDeviceResolution
        
        let faceBounds = VNImageRectForNormalizedRect(faceObservation.boundingBox, Int(displaySize.width), Int(displaySize.height))
        faceRectanglePath.addRect(faceBounds)
        
        if let landmarks = faceObservation.landmarks {
            // Landmarks are relative to -- and normalized within --- face bounds
            let affineTransform = CGAffineTransform(translationX: faceBounds.origin.x, y: faceBounds.origin.y)
                .scaledBy(x: faceBounds.size.width, y: faceBounds.size.height)
            
            // Treat eyebrows and lines as open-ended regions when drawing paths.
            let openLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEyebrow,
                landmarks.rightEyebrow,
                landmarks.faceContour,
                landmarks.noseCrest,
                landmarks.medianLine
            ]
            for openLandmarkRegion in openLandmarkRegions where openLandmarkRegion != nil {
                self.addPoints(in: openLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: false)
            }
            
            // Draw eyes, lips, and nose as closed regions.
            let closedLandmarkRegions: [VNFaceLandmarkRegion2D?] = [
                landmarks.leftEye,
                landmarks.rightEye,
                landmarks.outerLips,
                landmarks.innerLips,
                landmarks.nose
            ]
            for closedLandmarkRegion in closedLandmarkRegions where closedLandmarkRegion != nil {
                self.addPoints(in: closedLandmarkRegion!, to: faceLandmarksPath, applying: affineTransform, closingWhenComplete: true)
            }
        }
    }
    
    /// - Tag: DrawPaths
    fileprivate func drawFaceObservations(_ faceObservations: [VNFaceObservation]) {
        guard let faceRectangleShapeLayer = self.detectedFaceRectangleShapeLayer,
            let faceLandmarksShapeLayer = self.detectedFaceLandmarksShapeLayer
            else {
                return
        }
        
        CATransaction.begin()
        
        CATransaction.setValue(NSNumber(value: true), forKey: kCATransactionDisableActions)
        
        let faceRectanglePath = CGMutablePath()
        let faceLandmarksPath = CGMutablePath()
        
        for faceObservation in faceObservations {
            self.addIndicators(to: faceRectanglePath,
                               faceLandmarksPath: faceLandmarksPath,
                               for: faceObservation)
        }
        
        faceRectangleShapeLayer.path = faceRectanglePath
        faceLandmarksShapeLayer.path = faceLandmarksPath
        
        self.updateLayerGeometry()
        
        CATransaction.commit()
    }
    
    // MARK: AVCaptureVideoDataOutputSampleBufferDelegate
    /// - Tag: PerformRequests
    // Handle delegate method callback on receiving a sample buffer.
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        var requestHandlerOptions: [VNImageOption: AnyObject] = [:]
        
        let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil)
        if cameraIntrinsicData != nil {
            requestHandlerOptions[VNImageOption.cameraIntrinsics] = cameraIntrinsicData
        }
        
        guard let pixelBuffer = sampleBuffer.imageBuffer else {
            print("Failed to obtain a CVPixelBuffer for the current output frame.")
            return
        }
        
        let exifOrientation = self.exifOrientationForCurrentDeviceOrientation()
        
        let faceLandmarksRequest = VNDetectFaceLandmarksRequest(completionHandler: { (request, error) in
            
            if error != nil {
                print("FaceLandmarks error: \(String(describing: error)).")
            }
            
            guard let landmarksRequest = request as? VNDetectFaceLandmarksRequest,
                let results = landmarksRequest.results as? [VNFaceObservation] else {
                    return
            }
            
            var cnnSmiles = 0
            var netSmiles = 0
            
            for result in results {
                if let landmarks = result.landmarks, let points = landmarks.allPoints {
                    let p = points
                    
                    if p.normalizedPoints.count != 65 {
                        continue
                    }
                    
                    var normalizedArray = [Double]()
                    for i in 0...p.normalizedPoints.count-1{
                        normalizedArray.append(Double(p.normalizedPoints[i].x))
                        normalizedArray.append(Double(p.normalizedPoints[i].y))
                    }
                    
                    guard let mlMultiArray = try? MLMultiArray(shape:[130], dataType:MLMultiArrayDataType.double) else {
                        fatalError("Unexpected runtime error. MLMultiArray")
                    }
                    
                    for i in 0...normalizedArray.count-1{
                        mlMultiArray[i] = NSNumber(value: normalizedArray[i])
                    }
                    
                    let smileNetInput = SmileNetInput(_1: p.normalizedPoints[0].x.d, _2: p.normalizedPoints[0].y.d, _3: p.normalizedPoints[1].x.d, _4: p.normalizedPoints[1].y.d, _5: p.normalizedPoints[2].x.d, _6: p.normalizedPoints[2].y.d, _7: p.normalizedPoints[3].x.d, _8: p.normalizedPoints[3].y.d, _9: p.normalizedPoints[4].x.d, _10: p.normalizedPoints[4].y.d, _11: p.normalizedPoints[5].x.d, _12: p.normalizedPoints[5].y.d, _13: p.normalizedPoints[6].x.d, _14: p.normalizedPoints[6].y.d, _15: p.normalizedPoints[7].x.d, _16: p.normalizedPoints[7].y.d, _17: p.normalizedPoints[8].x.d, _18: p.normalizedPoints[8].y.d, _19: p.normalizedPoints[9].x.d, _20: p.normalizedPoints[9].y.d, _21: p.normalizedPoints[10].x.d, _22: p.normalizedPoints[10].y.d, _23: p.normalizedPoints[11].x.d, _24: p.normalizedPoints[11].y.d, _25: p.normalizedPoints[12].x.d, _26: p.normalizedPoints[12].y.d, _27: p.normalizedPoints[13].x.d, _28: p.normalizedPoints[13].y.d, _29: p.normalizedPoints[14].x.d, _30: p.normalizedPoints[14].y.d, _31: p.normalizedPoints[15].x.d, _32: p.normalizedPoints[15].y.d, _33: p.normalizedPoints[16].x.d, _34: p.normalizedPoints[16].y.d, _35: p.normalizedPoints[17].x.d, _36: p.normalizedPoints[17].y.d, _37: p.normalizedPoints[18].x.d, _38: p.normalizedPoints[18].y.d, _39: p.normalizedPoints[19].x.d, _40: p.normalizedPoints[19].y.d, _41: p.normalizedPoints[20].x.d, _42: p.normalizedPoints[20].y.d, _43: p.normalizedPoints[21].x.d, _44: p.normalizedPoints[21].y.d, _45: p.normalizedPoints[22].x.d, _46: p.normalizedPoints[22].y.d, _47: p.normalizedPoints[23].x.d, _48: p.normalizedPoints[23].y.d, _49: p.normalizedPoints[24].x.d, _50: p.normalizedPoints[24].y.d, _51: p.normalizedPoints[25].x.d, _52: p.normalizedPoints[25].y.d, _53: p.normalizedPoints[26].x.d, _54: p.normalizedPoints[26].y.d, _55: p.normalizedPoints[27].x.d, _56: p.normalizedPoints[27].y.d, _57: p.normalizedPoints[28].x.d, _58: p.normalizedPoints[28].y.d, _59: p.normalizedPoints[29].x.d, _60: p.normalizedPoints[29].y.d, _61: p.normalizedPoints[30].x.d, _62: p.normalizedPoints[30].y.d, _63: p.normalizedPoints[31].x.d, _64: p.normalizedPoints[31].y.d, _65: p.normalizedPoints[32].x.d, _66: p.normalizedPoints[32].y.d, _67: p.normalizedPoints[33].x.d, _68: p.normalizedPoints[33].y.d, _69: p.normalizedPoints[34].x.d, _70: p.normalizedPoints[34].y.d, _71: p.normalizedPoints[35].x.d, _72: p.normalizedPoints[35].y.d, _73: p.normalizedPoints[36].x.d, _74: p.normalizedPoints[36].y.d, _75: p.normalizedPoints[37].x.d, _76: p.normalizedPoints[37].y.d, _77: p.normalizedPoints[38].x.d, _78: p.normalizedPoints[38].y.d, _79: p.normalizedPoints[39].x.d, _80: p.normalizedPoints[39].y.d, _81: p.normalizedPoints[40].x.d, _82: p.normalizedPoints[40].y.d, _83: p.normalizedPoints[41].x.d, _84: p.normalizedPoints[41].y.d, _85: p.normalizedPoints[42].x.d, _86: p.normalizedPoints[42].y.d, _87: p.normalizedPoints[43].x.d, _88: p.normalizedPoints[43].y.d, _89: p.normalizedPoints[44].x.d, _90: p.normalizedPoints[44].y.d, _91: p.normalizedPoints[45].x.d, _92: p.normalizedPoints[45].y.d, _93: p.normalizedPoints[46].x.d, _94: p.normalizedPoints[46].y.d, _95: p.normalizedPoints[47].x.d, _96: p.normalizedPoints[47].y.d, _97: p.normalizedPoints[48].x.d, _98: p.normalizedPoints[48].y.d, _99: p.normalizedPoints[49].x.d, _100: p.normalizedPoints[49].y.d, _101: p.normalizedPoints[50].x.d, _102: p.normalizedPoints[50].y.d, _103: p.normalizedPoints[51].x.d, _104: p.normalizedPoints[51].y.d, _105: p.normalizedPoints[52].x.d, _106: p.normalizedPoints[52].y.d, _107: p.normalizedPoints[53].x.d, _108: p.normalizedPoints[53].y.d, _109: p.normalizedPoints[54].x.d, _110: p.normalizedPoints[54].y.d, _111: p.normalizedPoints[55].x.d, _112: p.normalizedPoints[55].y.d, _113: p.normalizedPoints[56].x.d, _114: p.normalizedPoints[56].y.d, _115: p.normalizedPoints[57].x.d, _116: p.normalizedPoints[57].y.d, _117: p.normalizedPoints[58].x.d, _118: p.normalizedPoints[58].y.d, _119: p.normalizedPoints[59].x.d, _120: p.normalizedPoints[59].y.d, _121: p.normalizedPoints[60].x.d, _122: p.normalizedPoints[60].y.d, _123: p.normalizedPoints[61].x.d, _124: p.normalizedPoints[61].y.d, _125: p.normalizedPoints[62].x.d, _126: p.normalizedPoints[62].y.d, _127: p.normalizedPoints[63].x.d, _128: p.normalizedPoints[63].y.d, _129: p.normalizedPoints[64].x.d, _130: p.normalizedPoints[64].y.d)
                    
                    let input = SmileCNNInput(input1: mlMultiArray)
                    
                    guard let smileOutputCNN = try? self.modelCNN.prediction(input: input) else {
                        fatalError("Unexpected runtime error.")
                    }
                    guard let smileOutputNet = try? self.modelNet.prediction(input: smileNetInput) else {
                        fatalError("Unexpected runtime error.")
                    }
                    
                    let value = smileOutputCNN.output1[0].doubleValue

                    if value > 0.5 {
                        cnnSmiles += 1
                    }
                    if smileOutputNet.y == 1 {
                        netSmiles += 1
                    }
                    
                    DispatchQueue.main.async {
                        //let formatter = NumberFormatter()
                        //formatter.maximumFractionDigits = 2
                        //formatter.minimumFractionDigits = 2
                        //let value = formatter.string(from: NSNumber(value: (smileOutput.yProbability[smileOutput.y] ?? 0) * 100))
                        // let percent = "(\(value ?? "0")%)"
                        
                        self.infoLabel.text = "CNN: \(cnnSmiles) 😃 \(results.count - cnnSmiles) 😐                                     NET:\(netSmiles) 😃 \(results.count - netSmiles) 😐"
                    }
                    
                }
            }
            
//            if results.count == smiles {
//                DispatchQueue.main.async {
//                    self.infoLabel.text = "📸 \(smiles) 😃 \(results.count - smiles) 😐"
//                }
//            }
            
            // Perform all UI updates (drawing) on the main queue, not the background queue on which this handler is being called.
            DispatchQueue.main.async {
                self.drawFaceObservations(results)
            }
        })
        
        do {
            try self.sequenceRequestHandler.perform([faceLandmarksRequest],
                                                    on: pixelBuffer,
                                                    orientation: exifOrientation)
        } catch let error as NSError {
            NSLog("Failed to perform SequenceRequest: %@", error)
        }
        
    }
}

extension CGFloat {
    
    var d: Double {
        return Double(self)
    }
}
