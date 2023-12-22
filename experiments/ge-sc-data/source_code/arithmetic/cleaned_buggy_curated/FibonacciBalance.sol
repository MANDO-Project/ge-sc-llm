pragma solidity ^0.4.0;
contract FibonacciBalance {
    address public fibonacciLibrary;
    uint public calculatedFibNumber;
    uint public start = 3;
    uint public withdrawalCounter;
    bytes4 constant fibSig = bytes4(sha3("setFibonacci(uint256)"));
    constructor(address _fibonacciLibrary) public payable {
        fibonacciLibrary = _fibonacciLibrary;
    }
    function withdraw() {
        withdrawalCounter += 1;
        require(fibonacciLibrary.delegatecall(fibSig, withdrawalCounter));
        msg.sender.transfer(calculatedFibNumber * 1 ether);
    }
    function() public {
        require(fibonacciLibrary.delegatecall(msg.data));
    }
}
contract FibonacciLib {
    uint public start;
    uint public calculatedFibNumber;
    function setStart(uint _start) public {
        start = _start;
    }
    function setFibonacci(uint n) public {
        calculatedFibNumber = fibonacci(n);
    }
    function fibonacci(uint n) internal returns (uint) {
        if (n == 0) return start;
        else if (n == 1) return start + 1;
        else return fibonacci(n - 1) + fibonacci(n - 2);
    }
}