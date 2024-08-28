//
//  DataStreamCellTableViewCell.swift
//  catProject
//
//  Created by Vaishnavi Purusothaman on 31/05/24.
//

import UIKit

class DataStreamCellTableViewCell: UITableViewCell {

    @IBOutlet weak var idLbl: UILabel!
    
    @IBOutlet weak var statusLbl: UILabel!
    
    @IBOutlet weak var valueLbl: UILabel!
    
    override func awakeFromNib() {
        super.awakeFromNib()
        // Initialization code
    }

    override func setSelected(_ selected: Bool, animated: Bool) {
        super.setSelected(selected, animated: animated)

        // Configure the view for the selected state
    }
    
}
