pragma solidity ^0.4.0;
contract SimpleSuicide {
  function sudicideAnyone() {
    selfdestruct(msg.sender);
  }
}