pragma solidity >=0.5.1;
contract owned {
  function bug_tmstmp21() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  address public owner;
    constructor() public {
        owner = msg.sender;
    }
address winner_tmstmp3;
function play_tmstmp3(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp3 = msg.sender;}}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address newOwner) onlyOwner public {
        owner = newOwner;
    }
function bug_tmstmp9() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
}
contract tokenRecipient {
  address winner_tmstmp27;
function play_tmstmp27(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp27 = msg.sender;}}
  event receivedEther(address sender, uint amount);
  address winner_tmstmp31;
function play_tmstmp31(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp31 = msg.sender;}}
  event receivedTokens(address _from, uint256 _value, address _token, bytes _extraData);
    function receiveApproval(address _from, uint256 _value, address _token, bytes memory _extraData) public {
        Token t = Token(_token);
        require(t.transferFrom(_from, address(this), _value));
        emit receivedTokens(_from, _value, _token, _extraData);
    }
function bug_tmstmp25() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
    function () payable external {
        emit receivedEther(msg.sender, msg.value);
    }
address winner_tmstmp19;
function play_tmstmp19(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp19 = msg.sender;}}
}
contract Token {
    function totalSupply() public view returns (uint256);
address winner_tmstmp26;
function play_tmstmp26(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp26 = msg.sender;}}
    function actualBalanceOf(address _owner) public view returns (uint256 balance);
function bug_tmstmp20 () public payable {
	uint pastBlockTime_tmstmp20;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp20);  
        pastBlockTime_tmstmp20 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function bug_tmstmp32 () public payable {
	uint pastBlockTime_tmstmp32;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp32);  
        pastBlockTime_tmstmp32 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function renounceOwnership() public;
address winner_tmstmp38;
function play_tmstmp38(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp38 = msg.sender;}}
    function transferOwnership(address _newOwner) public;
function bug_tmstmp4 () public payable {
	uint pastBlockTime_tmstmp4;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp4);  
        pastBlockTime_tmstmp4 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function pause() public;
address winner_tmstmp7;
function play_tmstmp7(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp7 = msg.sender;}}
    function unpause() public;
address winner_tmstmp23;
function play_tmstmp23(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp23 = msg.sender;}}
}
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "Safe mul error");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "Safe div error");
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "Safe sub error");
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "Safe add error");
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "Safe mod error");
        return a % b;
    }
}
contract MindsyncPlatform is owned, tokenRecipient {
    using SafeMath for uint256;
  address winner_tmstmp10;
function play_tmstmp10(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp10 = msg.sender;}}
  uint public minimumQuorum;
  address winner_tmstmp22;
function play_tmstmp22(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp22 = msg.sender;}}
  uint public minimumTokensToVote;
  function bug_tmstmp12 () public payable {
	uint pastBlockTime_tmstmp12;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp12);  
        pastBlockTime_tmstmp12 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
  uint public debatingPeriodInMinutes;
  address winner_tmstmp11;
function play_tmstmp11(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp11 = msg.sender;}}
  Proposal[] public proposals;
  function bug_tmstmp1() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  uint public numProposals;
  address winner_tmstmp2;
function play_tmstmp2(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp2 = msg.sender;}}
  Token public tokenAddress;
  function bug_tmstmp17() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  address chairmanAddress;
  function bug_tmstmp37() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  bool public initialized = false;
  function bug_tmstmp13() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
  event Initialized();
  uint256 bugv_tmstmp5 = block.timestamp;
  event ProposalAdded(uint proposalID, address recipient, uint amount, string description);
  uint256 bugv_tmstmp1 = block.timestamp;
  event Voted(uint proposalID, bool position, address voter);
  uint256 bugv_tmstmp2 = block.timestamp;
  event ProposalTallied(uint proposalID, uint result, uint quorum, bool active);
  uint256 bugv_tmstmp3 = block.timestamp;
  event ChangeOfRules(uint newMinimumTokensToVote, uint newMinimumQuorum, uint newDebatingPeriodInMinutes, address newTokenAddress, address newChairmanAddress);
  uint256 bugv_tmstmp4 = block.timestamp;
  event ProposalSignedByChairman(uint proposalNumber,  bool sign, address chairman);
    struct Proposal {
        address recipient;
        uint amount;
        string description;
        bool signedByChairman;
        uint minExecutionDate;
        bool executed;
        bool proposalPassed;
        uint numberOfVotes;
        bytes32 proposalHash;
        Vote[] votes;
        mapping (address => bool) voted;
    }
    struct Vote {
        bool inSupport;
        address voter;
    }
    modifier onlyTokenholders {
        require(tokenAddress.actualBalanceOf(msg.sender) > minimumTokensToVote);
        _;
    }
    modifier onlyChairman {
        require(msg.sender == chairmanAddress);
        _;
    }
    constructor() payable public {
    }
address winner_tmstmp14;
function play_tmstmp14(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp14 = msg.sender;}}
    function init(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(!initialized);
        initialized = true;
        changeVotingRules(_tokenAddress, _chairmanAddress, _minimumTokensToVote, _minimumPercentToPassAVote, _minutesForDebate);
        emit Initialized();
    }
address winner_tmstmp30;
function play_tmstmp30(uint startTime) public {
	if (startTime + (5 * 1 days) == block.timestamp){
		winner_tmstmp30 = msg.sender;}}
    function changeVotingRules(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(_chairmanAddress != address(0));
        require(_minimumPercentToPassAVote <= 51);
        tokenAddress = Token(_tokenAddress);
        chairmanAddress = _chairmanAddress;
        if (_minimumTokensToVote == 0 ) _minimumTokensToVote = 1;
        minimumTokensToVote = _minimumTokensToVote;
        if (_minimumPercentToPassAVote == 0 ) _minimumPercentToPassAVote = 51;
        minimumQuorum = _minimumPercentToPassAVote;
        debatingPeriodInMinutes = _minutesForDebate;
        emit ChangeOfRules(_minimumTokensToVote, minimumQuorum, debatingPeriodInMinutes, address(tokenAddress), chairmanAddress);
    }
function bug_tmstmp8 () public payable {
	uint pastBlockTime_tmstmp8;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp8);  
        pastBlockTime_tmstmp8 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function newProposal(
        address destination,
        uint weiAmount,
        string memory transactionDescription,
        bytes memory transactionBytecode
    )
        onlyTokenholders public
        returns (uint proposalID)
    {
        proposalID = proposals.length++;
        Proposal storage p = proposals[proposalID];
        p.recipient = destination;
        p.signedByChairman = false;
        p.amount = weiAmount;
        p.description = transactionDescription;
        p.proposalHash = keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
        p.minExecutionDate = now + debatingPeriodInMinutes * 1 minutes;
        p.executed = false;
        p.proposalPassed = false;
        p.numberOfVotes = 0;
        emit ProposalAdded(proposalID, destination, weiAmount, transactionDescription);
        numProposals = proposalID+1;
        return proposalID;
    }
address winner_tmstmp39;
function play_tmstmp39(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp39 = msg.sender;}}
    function checkProposalCode(
        uint proposalNumber,
        address destination,
        uint weiAmount,
        bytes memory transactionBytecode
    )
        view public
        returns (bool codeChecksOut)
    {
        Proposal storage p = proposals[proposalNumber];
        return p.proposalHash == keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
    }
function bug_tmstmp36 () public payable {
	uint pastBlockTime_tmstmp36;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp36);  
        pastBlockTime_tmstmp36 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function sign(
        uint proposalNumber,
        bool signProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        require(initialized);
        Proposal storage p = proposals[proposalNumber];
        require(msg.sender == chairmanAddress);
        require(signProposal == true);
        p.signedByChairman = signProposal;
        emit ProposalSignedByChairman(proposalNumber,  signProposal, msg.sender);
        return proposalNumber;
    }
address winner_tmstmp35;
function play_tmstmp35(uint startTime) public {
	uint _vtime = block.timestamp;
	if (startTime + (5 * 1 days) == _vtime){
		winner_tmstmp35 = msg.sender;}}
    function vote(
        uint proposalNumber,
        bool supportsProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        Proposal storage p = proposals[proposalNumber];
        require(p.voted[msg.sender] != true);
        voteID = p.votes.length++;
        p.votes[voteID] = Vote({inSupport: supportsProposal, voter: msg.sender});
        p.voted[msg.sender] = true;
        p.numberOfVotes = voteID +1;
        emit Voted(proposalNumber,  supportsProposal, msg.sender);
        return voteID;
    }
function bug_tmstmp40 () public payable {
	uint pastBlockTime_tmstmp40;  
	require(msg.value == 10 ether);  
        require(now != pastBlockTime_tmstmp40);  
        pastBlockTime_tmstmp40 = now;        
        if(now % 15 == 0) {  
            msg.sender.transfer(address(this).balance);
        }
    }
    function executeProposal(uint proposalNumber, bytes memory transactionBytecode) public {
        Proposal storage p = proposals[proposalNumber];
        require(initialized);
        require(now > p.minExecutionDate                                              
            && !p.executed                                                           
            && p.proposalHash == keccak256(abi.encodePacked(p.recipient, p.amount, transactionBytecode)));  
        uint quorum = 0;
        uint yea = 0;
        uint nay = 0;
        for (uint i = 0; i <  p.votes.length; ++i) {
            Vote storage v = p.votes[i];
            uint voteWeight = tokenAddress.actualBalanceOf(v.voter);
            quorum += voteWeight;
            if (v.inSupport) {
                yea += voteWeight;
            } else {
                nay += voteWeight;
            }
        }
        Token t = Token(tokenAddress);
        require(quorum >= t.totalSupply().mul(minimumQuorum).div(100));  
        if (yea > nay ) {
            p.executed = true;
            (bool success, ) = p.recipient.call.value(p.amount)(transactionBytecode);
            require(success);
            p.proposalPassed = true;
        } else {
            p.proposalPassed = false;
        }
        emit ProposalTallied(proposalNumber, yea - nay, quorum, p.proposalPassed);
    }
function bug_tmstmp33() view public returns (bool) {
    return block.timestamp >= 1546300800;
  }
}