//
//  ViewController.swift
//  catProject
//
//  Created by Vaishnavi Purusothaman on 29/05/24.
//

import UIKit

struct DataStream: Codable {
    let id: String
    let current_value: String
    let at: String
}

struct Body: Codable {
    let id: String
    let datastreams: [DataStream]
    let uuid: String
}

struct Response: Codable {
    let body: Body
    let resource: String
}


class WebSocketManager {
    private var webSocketTask: URLSessionWebSocketTask?
    private let urlSession: URLSession
    var onMessageReceived: (([String: Any]) -> Void)?

    init() {
        self.urlSession = URLSession(configuration: .default)
    }

    func connect() {
        guard let url = URL(string: "wss://demo.sewio.net") else {
            print("Invalid URL")
            return
        }

        var request = URLRequest(url: url)
        request.setValue("171555a8fe71148a165392904", forHTTPHeaderField: "X-ApiKey")

        webSocketTask = urlSession.webSocketTask(with: request)
        webSocketTask?.resume()

        receiveMessage()
        sendSubscribeMessage()
    }

    func disconnect() {
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            switch result {
            case .failure(let error):
                print("Error in receiving message: \(error)")
            case .success(let message):
                switch message {
                case .string(let text):
                    self?.handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        self?.handleMessage(text)
                    }
                @unknown default:
                    fatalError()
                }
            }
            self?.receiveMessage()
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
//                print(json)
                onMessageReceived?(json)
            }
        } catch {
            print("Failed to parse JSON: \(error)")
        }
    }

    private func sendSubscribeMessage() {
        let subscribeMessage: [String: Any] = [
            "headers": ["X-ApiKey": "171555a8fe71148a165392904"],
            "method": "subscribe",
            "resource": "/feeds/14"
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: subscribeMessage, options: [])
            let message = URLSessionWebSocketTask.Message.data(jsonData)
            webSocketTask?.send(message) { error in
                if let error = error {
                    print("Error sending message: \(error)")
                } else {
                    print("Subscribe message sent")
                }
            }
        } catch {
            print("Failed to serialize subscribe message: \(error)")
        }
    }
}

import UIKit

class ViewController: UIViewController,UITableViewDelegate,UITableViewDataSource {
    private let webSocketManager = WebSocketManager()
    private var imageView: UIImageView!
    private var movableView: UIView!
    
    @IBOutlet weak var backImageView: UIImageView!
    @IBOutlet weak var backView: UIView!
    
    @IBOutlet weak var tableView: UITableView!
    
    var dataStreams: [DataStream] = []
    
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        tableView.dataSource = self
        tableView.register(UINib(nibName: "DataStreamCellTableViewCell", bundle: nil), forCellReuseIdentifier: "DataStreamCellTableViewCell")
        setupImageView()
        setupMovableView()
        setupWebSocketManager()
        
        
        
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        webSocketManager.disconnect()
    }
    
    private func setupImageView() {
        backView.backgroundColor = UIColor.yellow
        
        
    }
    
    private func setupMovableView() {
        let size: CGFloat = 10
        movableView = UIView(frame: CGRect(x: 0, y: 0, width: 10, height: 10))
        movableView.backgroundColor = .red
        movableView.layer.cornerRadius = size / 2
        movableView.clipsToBounds = true
        backImageView.addSubview(movableView)
    }
    
    private func setupWebSocketManager() {
        webSocketManager.connect()
        webSocketManager.onMessageReceived = { [weak self] json in
            self?.handleWebSocketMessage(json)
        }
    }
    
    private func handleWebSocketMessage(_ json: [String: Any]) {
        guard let body = json["body"] as? [String: Any],
              let datastreams = body["datastreams"] as? [[String: Any]] else { return }
        
        var newDataStreams: [DataStream] = []
        
        for datastream in datastreams {
            if let id = datastream["id"] as? String,
               let valueString = datastream["current_value"] as? String,
               let at = datastream["at"] as? String {
                let newStream = DataStream(id: id, current_value: valueString, at: at)
                newDataStreams.append(newStream)
            }
        }
        
        DispatchQueue.main.async {
            self.dataStreams = newDataStreams
            self.tableView.reloadData()
        }
        
        
        var posX: CGFloat?
        var posY: CGFloat?
        
        for datastream in datastreams {
            if let id = datastream["id"] as? String,
               let valueString = datastream["current_value"] as? String,
               let value = Double(valueString) {
                switch id {
                case "posX":
                    posX = CGFloat(value)
                case "posY":
                    posY = CGFloat(value)
                default:
                    break
                }
            }
        }
        
        DispatchQueue.main.async {
            if let x = posX, let y = posY {
                
                let imagewidth = self.backImageView.frame.size.width
                let imageheight = self.backImageView.frame.size.height
                var floorPlanX: CGFloat = 44.00;
                var floorPlanY: CGFloat = 33.11;
                
                floorPlanX = imagewidth/floorPlanX
                floorPlanY = imageheight/floorPlanY
                
                self.updateViewPosition(x: x*floorPlanX, y: y*floorPlanY)
            }
        }
    }
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return dataStreams.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "DataStreamCellTableViewCell", for: indexPath) as! DataStreamCellTableViewCell
        let dataStream = dataStreams[indexPath.row]
        cell.idLbl.text = dataStream.id
        cell.valueLbl.text = "Value: \(dataStream.current_value)"
        cell.statusLbl.text = "At: \(dataStream.at)"
        return cell
    }
    
    private func updateViewPosition(x: CGFloat, y: CGFloat) {
        // Adjusting the position relative to the imageView's bounds
        
        let adjustedX = min(max(x, 0), backImageView.bounds.width - movableView.bounds.width)
        let adjustedY = min(max(y, 0), backImageView.bounds.height - movableView.bounds.height)
        print(adjustedX)
        print(adjustedY)
        movableView.frame.origin = CGPoint(x: adjustedX, y: adjustedY)
    }
}


