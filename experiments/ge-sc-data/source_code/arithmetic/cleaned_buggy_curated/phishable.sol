 pragma solidity ^0.4.22;
 contract Phishable {
    address public owner;
    constructor (address _owner) {
        owner = _owner;
    }
    function () public payable {}  
    function withdrawAll(address _recipient) public {
        require(tx.origin == owner);
        _recipient.transfer(this.balance);
    }
}