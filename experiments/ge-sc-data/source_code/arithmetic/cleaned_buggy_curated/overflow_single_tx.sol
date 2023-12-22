pragma solidity ^0.4.23;
contract IntegerOverflowSingleTransaction {
    uint public count = 1;
    function overflowaddtostate(uint256 input) public {
        count += input;
    }
    function overflowmultostate(uint256 input) public {
        count *= input;
    }
    function underflowtostate(uint256 input) public {
        count -= input;
    }
    function overflowlocalonly(uint256 input) public {
        uint res = count + input;
    }
    function overflowmulocalonly(uint256 input) public {
        uint res = count * input;
    }
    function underflowlocalonly(uint256 input) public {
       	uint res = count - input;
    }
}