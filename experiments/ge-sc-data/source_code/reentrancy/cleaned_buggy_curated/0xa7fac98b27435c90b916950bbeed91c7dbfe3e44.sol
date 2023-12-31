pragma solidity ^0.4.18;
contract ERC20Basic {
  function totalSupply() public view returns (uint256);
  function balanceOf(address who) public view returns (uint256);
  function transfer(address to, uint256 value) public returns (bool);
  event Transfer(address indexed from, address indexed to, uint256 value);
}
contract ERC20 is ERC20Basic {
  function allowance(address owner, address spender) public view returns (uint256);
  function transferFrom(address from, address to, uint256 value) public returns (bool);
  function approve(address spender, uint256 value) public returns (bool);
  event Approval(address indexed owner, address indexed spender, uint256 value);
}
library SafeERC20 {
  function safeTransfer(ERC20Basic token, address to, uint256 value) internal {
    assert(token.transfer(to, value));
  }
  function safeTransferFrom(ERC20 token, address from, address to, uint256 value) internal {
    assert(token.transferFrom(from, to, value));
  }
  function safeApprove(ERC20 token, address spender, uint256 value) internal {
    assert(token.approve(spender, value));
  }
}
contract TokenTimelock {
  using SafeERC20 for ERC20Basic;
  ERC20Basic public token;
  address public beneficiary;
  uint256 public releaseTime;
  function TokenTimelock(ERC20Basic _token, address _beneficiary, uint256 _releaseTime) public {
    require(_releaseTime > now);
    token = _token;
    beneficiary = _beneficiary;
    releaseTime = _releaseTime;
  }
  function release() public {
    require(now >= releaseTime);
    uint256 amount = token.balanceOf(this);
    require(amount > 0);
    token.safeTransfer(beneficiary, amount);
  }
}
contract ZipperTokenTimelockFactoryMonthLockup {
   function create(ERC20Basic _token, address _beneficiary) public
   {
       emit Created(new TokenTimelock(_token, _beneficiary, now + 180 days));
   }
   event Created(TokenTimelock _address);
}