// based on:   https://github.com/fogleman/GrayScott

import Cocoa
import MetalKit

let XSIZE:Int = 300
let YSIZE:Int = XSIZE
let NUMCELLS:Int = XSIZE * YSIZE

let feedRateMin:Float = 0.02
let feedRateMax:Float = 0.05
let killRateMin:Float = 0.04
let killRateMax:Float = 0.08
let diffusionRateAMin:Float = 0.8
let diffusionRateAMax:Float = 1.3
let diffusionRateBMin:Float = 0.2
let diffusionRateBMax:Float = 0.6
let scaleMin:Float = 0.5
let scaleMax:Float = 1.5

class ViewController: NSViewController, NSWindowDelegate {
    var ctrl = Control()
    var cellsA:[Cell] = Array(repeating: Cell(), count: NUMCELLS)
    var cellsB:[Cell] = Array(repeating: Cell(), count: NUMCELLS)
    
    var controlBuffer:MTLBuffer! = nil
    var cellsBufferA0:MTLBuffer! = nil
    var cellsBufferA1:MTLBuffer! = nil
    var cellsBufferB0:MTLBuffer! = nil
    var cellsBufferB1:MTLBuffer! = nil
    var colorBuffer  :MTLBuffer! = nil
    let ctrlBufferSize:Int = MemoryLayout<Control>.stride
    let cellsBufferSize:Int = MemoryLayout<Cell>.stride * NUMCELLS
    
    lazy var device2D: MTLDevice! = MTLCreateSystemDefaultDevice()
    lazy var commandQueue: MTLCommandQueue! = { return self.device2D.makeCommandQueue() }()
    var threadGroupCount = MTLSize()
    var threadGroups = MTLSize()
    var texture: MTLTexture!
    var pipeline:[MTLComputePipelineState] = []
    let queue = DispatchQueue(label:"Q")
    
    @IBOutlet var metalTextureView: MetalTextureView!
    @IBOutlet var s1: NSSlider!
    @IBOutlet var s2: NSSlider!
    @IBOutlet var s3: NSSlider!
    @IBOutlet var s4: NSSlider!
    @IBOutlet var s5: NSSlider!
    
    @IBAction func sliderChanged(_ sender: NSSlider) {
        switch sender {
        case s1 : ctrl.feedRate = feedRateMin + (feedRateMax - feedRateMin) * sender.floatValue
        case s2 : ctrl.killRate = killRateMin + (killRateMax - killRateMin) * sender.floatValue
        case s3 : ctrl.diffusionRateA = diffusionRateAMin + (diffusionRateAMax - diffusionRateAMin) * sender.floatValue
        case s4 : ctrl.diffusionRateB = diffusionRateBMin + (diffusionRateBMax - diffusionRateBMin) * sender.floatValue
        case s5 : ctrl.scale = scaleMin + (scaleMax - scaleMin) * sender.floatValue
        default : break
        }
        
        reset()
    }
    
    func encode(_ slider:NSSlider, _ value:Float, _ vMin:Float, _ vMax:Float) {
        let v = min(max(value,vMin),vMax)
        slider.floatValue = (v - vMin) / (vMax - vMin)
        sliderChanged(slider)
    }
    
    func alter(_ slider:NSSlider, _ dir:Float) {
        slider.floatValue = min(max(slider.floatValue + dir * 0.005,0),1)
        sliderChanged(slider)
    }
    
    let PIPELINE_DRAW = 0
    let PIPELINE_EVOLVE = 1
    let shaderNames = [ "drawShader","evolveShader" ]
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.view.wantsLayer = true
        
        controlBuffer = device2D.makeBuffer(length:ctrlBufferSize,  options:MTLResourceOptions.storageModeShared)
        cellsBufferA0 = device2D.makeBuffer(length:cellsBufferSize, options:MTLResourceOptions.storageModeShared)
        cellsBufferA1 = device2D.makeBuffer(length:cellsBufferSize, options:MTLResourceOptions.storageModeShared)
        cellsBufferB0 = device2D.makeBuffer(length:cellsBufferSize, options:MTLResourceOptions.storageModeShared)
        cellsBufferB1 = device2D.makeBuffer(length:cellsBufferSize, options:MTLResourceOptions.storageModeShared)
        
        let jbSize = MemoryLayout<simd_float3>.stride * 256
        colorBuffer = device2D.makeBuffer(length:jbSize, options:MTLResourceOptions.storageModeShared)
        colorBuffer.contents().copyMemory(from:colorMap, byteCount:jbSize)
        
        let defaultLibrary:MTLLibrary! = device2D.makeDefaultLibrary()
        
        //------------------------------
        func loadShader(_ name:String) -> MTLComputePipelineState {
            do {
                guard let fn = defaultLibrary.makeFunction(name: name)  else { print("shader not found: " + name); exit(0) }
                return try device2D.makeComputePipelineState(function: fn)
            }
            catch { print("pipeline failure for : " + name); exit(0) }
        }
        
        for i in 0 ..< shaderNames.count { pipeline.append(loadShader(shaderNames[i])) }
        //------------------------------
        
        let w = pipeline[PIPELINE_EVOLVE].threadExecutionWidth
        let h = pipeline[PIPELINE_EVOLVE].maxTotalThreadsPerThreadgroup / w
        
        threadGroupCount = MTLSizeMake(w, h, 1)
        threadGroups = MTLSizeMake(1 + (XSIZE / w), 1 + (YSIZE / h), 1)
        
        ctrl.cxSize = Int32(XSIZE)
        ctrl.cySize = Int32(YSIZE)
        ctrl.numCells = Int32(NUMCELLS)
    }
    
    override func viewDidAppear() {
        self.view.layer?.backgroundColor = NSColor.darkGray.cgColor
        
        view.window?.delegate = self    // so we receive window size changed notifications
        resizeIfNecessary()
        dvrCount = 1 // resize metalview without delay
        
        encode(s1,0.039,    feedRateMin,feedRateMax)
        encode(s2,0.069,    killRateMin,killRateMax)
        encode(s3,1.1,      diffusionRateAMin,diffusionRateAMax)
        encode(s4,0.4,      diffusionRateBMin,diffusionRateBMax)
        encode(s5,1,        scaleMin,scaleMax)
        
        layoutViews()
        reset()
        
        Timer.scheduledTimer(withTimeInterval:0.01, repeats:true) { timer in self.timerHandler() }
    }
    
    //MARK: -
    
    var dvrCount:Int = 0
    
    func windowDidResize(_ notification: Notification) {
        dvrCount = 10 // 20 = 1 second delay
        resizeIfNecessary()
    }
    
    func resizeIfNecessary() {
        let minWinSize:CGSize = CGSize(width:780, height:800)
        var r:CGRect = (view.window?.frame)!
        var needSizing:Bool = false
        
        if r.size.width  < minWinSize.width  { r.size.width = minWinSize.width;     needSizing = true }
        if r.size.height < minWinSize.height { r.size.height = minWinSize.height;   needSizing = true }
        
        if needSizing { view.window?.setFrame(r, display: true) }
    }
    
    //MARK: -
    
    func layoutViews() {
        ctrl.txSize = Int32(metalTextureView.frame.width)
        ctrl.tySize = Int32(metalTextureView.frame.height)
        ctrl.zoom = 1 + Int32( min(Float(ctrl.txSize)/Float(ctrl.cxSize),Float(ctrl.tySize)/Float(ctrl.cySize)))
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: Int(ctrl.txSize),
            height: Int(ctrl.tySize),
            mipmapped: false)
        
        texture = device2D.makeTexture(descriptor: textureDescriptor)!
        metalTextureView.initialize(texture)
        
        let xs = view.bounds.width
        let ys = view.bounds.height
        metalTextureView.frame = CGRect(x:0, y:55, width:xs, height:ys-55)
        reset()
    }
    
    //MARK: -
    
    func reset() {
        for i in 0 ..< NUMCELLS {
            cellsA[i].value = 1
            cellsB[i].value = drand48() < 0.01 ? 1 : 0
        }
        
        cellsBufferA0.contents().copyMemory(from:cellsA, byteCount:cellsBufferSize)
        cellsBufferB0.contents().copyMemory(from:cellsB, byteCount:cellsBufferSize)
    }
    
    //MARK: -
    
    @objc func timerHandler() {
        if dvrCount > 0 {
            dvrCount -= 1
            if dvrCount <= 0 {
                layoutViews()
            }
        }
        
        evolve()
        draw()
    }
    
    func fRandom() -> Float {
        var f = Float(drand48())
        if(f == 1.0) { f = 0.99 }  // disallow pure white for this demo
        return f
    }
    
    func randomColor() -> simd_float4 { return simd_float4(fRandom(),fRandom(),fRandom(),1)  }
    
    //MARK: -
    
    func evolve() {
        func evolveCycle(_ atob:Bool) {
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
            commandEncoder.setComputePipelineState(pipeline[PIPELINE_EVOLVE])
            
            commandEncoder.setBuffer(controlBuffer, offset: 0, index: 0)
            commandEncoder.setBuffer(cellsBufferA0, offset: 0, index: atob ? 3 : 1)
            commandEncoder.setBuffer(cellsBufferB0, offset: 0, index: atob ? 4 : 2)
            commandEncoder.setBuffer(cellsBufferA1, offset: 0, index: atob ? 1 : 3)
            commandEncoder.setBuffer(cellsBufferB1, offset: 0, index: atob ? 2 : 4)
            
            commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
            commandEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        evolveCycle(false)
        evolveCycle(true)
    }
    
    //MARK: -
    
    func draw() {
        controlBuffer.contents().copyMemory(from: &ctrl, byteCount:ctrlBufferSize)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        
        commandEncoder.setComputePipelineState(pipeline[PIPELINE_DRAW])
        
        commandEncoder.setTexture(texture, index: 0)
        commandEncoder.setBuffer(controlBuffer, offset: 0, index: 0)
        commandEncoder.setBuffer(cellsBufferB0, offset: 0, index: 1)
        commandEncoder.setBuffer(colorBuffer,   offset: 0, index: 2)
        
        commandEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupCount)
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        metalTextureView.display(metalTextureView.layer!)
    }
    
    //MARK: -
    
    override func keyDown(with event: NSEvent) {
        let keyCode = event.charactersIgnoringModifiers!.uppercased()
        //print("KeyDown ",keyCode,event.keyCode)
        
        switch keyCode {
        case "R" : reset()
        case "1" : alter(s1,-1)
        case "2" : alter(s1,+1)
        case "3" : alter(s2,-1)
        case "4" : alter(s2,+1)
        case "5" : alter(s3,-1)
        case "6" : alter(s3,+1)
        case "7" : alter(s4,-1)
        case "8" : alter(s4,+1)
        case "9" : alter(s5,-1)
        case "0" : alter(s5,+1)
        default  : break
        }
    }
    
    //    //MARK: -
    //
    //    func flippedYCoord(_ pt:NSPoint) -> NSPoint {
    //        var npt = pt
    //        npt.y = view.bounds.size.height - pt.y
    //        return npt
    //    }
    //
    //    var pt1 = NSPoint()
    //    var pt2 = NSPoint()
    //
    //    override func mouseDown(with event: NSEvent) {
    //        //        pt1 = pt2
    //        //        pt2 = flippedYCoord(event.locationInWindow)
    //    }
    //
    //    override func mouseDragged(with event: NSEvent) {}
    //    override func mouseUp(with event: NSEvent) {}
    //    override func rightMouseDown(with event: NSEvent) {}
    //    override func rightMouseDragged(with event: NSEvent) {}
    //    override func scrollWheel(with event: NSEvent) {}
}

// ===============================================

class BaseNSView: NSView {
    override var acceptsFirstResponder: Bool { return true }
}

