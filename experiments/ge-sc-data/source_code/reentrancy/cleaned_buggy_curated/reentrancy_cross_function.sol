pragma solidity ^0.4.0;
contract Reentrancy_cross_function {
    mapping (address => uint) private userBalances;
    function transfer(address to, uint amount) {
        if (userBalances[msg.sender] >= amount) {
            userBalances[to] += amount;
            userBalances[msg.sender] -= amount;
        }
    }
    function withdrawBalance() public {
        uint amountToWithdraw = userBalances[msg.sender];
        (bool success, ) = msg.sender.call.value(amountToWithdraw)("");  
        require(success);
        userBalances[msg.sender] = 0;
    }
}